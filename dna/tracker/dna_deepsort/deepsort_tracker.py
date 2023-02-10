
from typing import List, Union, Tuple, Any
from datetime import datetime
from pathlib import Path
import sys
from enum import Enum

import numpy as np
import cv2
from omegaconf.omegaconf import OmegaConf
import shapely.geometry as geometry

from dna import Frame, Image, BGR, color, plot_utils
from dna.tracker import DNASORTParams

FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0] / 'deepsort')
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

import dna
from dna import Box, Size2d
from dna.detect import ObjectDetector, Detection
from dna.tracker import IDNATrack, TrackState, IouDistThreshold
from ..tracker import DetectionBasedObjectTracker
from .deepsort.deepsort import deepsort_rbc
from .deepsort.track import Track as DSTrack
from .deepsort.track import TrackState as DSTrackState

import logging
LOGGER = logging.getLogger('dna.tracker.dnasort')

DEFAULT_DETECTION_THRESHOLD = 0.35
DEFAULT_DETECTION_MIN_SIZE = Size2d(20, 15)
DEFAULT_DETECTION_MAX_SIZE = Size2d(768, 768)
DEFAULT_DETECTIION_MAPPING = {'car':'car', 'bus':'car', 'truck':'car'}

DEFAULT_IOU_DIST_THRESHOLD_TIGHT = IouDistThreshold(0.2, 7)
DEFAULT_IOU_DIST_THRESHOLD = IouDistThreshold(0.80, 70)
DEFAULT_IOU_DIST_THRESHOLD_LOOSE = IouDistThreshold(0.85, 90)
DEFAULT_IOU_DIST_THRESHOLD_GATE = IouDistThreshold(0.85, 210)

DEFAULT_METRIC_TIGHT_THRESHOLD = 0.3
DEFAULT_METRIC_THRESHOLD = 0.55
DEFAULT_METRIC_GATE_DISTANCE = 500
DEFAULT_METRIC_GATE_BOX_DISTANCE = 200
DEFAULT_METRIC_MIN_DETECTION_SIZE = Size2d(40, 35)
DEFAULT_MAX_FEATURE_COUNT = 50

DEFAULT_N_INIT = 3
DEFAULT_MAX_AGE = 10
DEFAULT_MIN_NEW_TRACK_SIZE = [30, 20]

DEFAULT_BLIND_ZONES = []
DEFAULT_EXIT_ZONES = []
DEFAULT_STABLE_ZONES = []
DEFAULT_OVERLAP_SUPRESS_RATIO = 0.75

    
class DeepSORTTrack(IDNATrack):
    from .deepsort.track import Track as DSTrack
    def __init__(self, ds_track: DSTrack, frame_index:int, timestamp:float) -> None:
        super().__init__()
        self.ds_track:DSTrack = ds_track
        self.__id = ds_track.track_id
        self.__location = Box.from_tlbr(np.rint(ds_track.to_tlbr()))
        self.__frame_index = frame_index
        self.__timestamp = timestamp
        
        if ds_track.state == DSTrackState.Confirmed:
            self.__state = TrackState.Confirmed if ds_track.time_since_update <= 0 else TrackState.TemporarilyLost
        elif ds_track.state == DSTrackState.Tentative:
            self.__state = TrackState.Tentative
        elif ds_track.state == DSTrackState.Deleted:
            self.__state = TrackState.Deleted
        
    @property
    def id(self) -> Any:
        return self.__id

    @property
    def state(self) -> TrackState:
        return self.__state

    @property
    def location(self) -> Box:
        return self.__location

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    @property
    def timestamp(self) -> float:
        return self.__timestamp
    
    def draw(self, convas:Image, color:BGR, label_color:BGR=None, line_thickness:int=2) -> Image:
        loc = self.location
        convas = loc.draw(convas, color, line_thickness=line_thickness)
        convas = cv2.circle(convas, loc.center().xy.astype(int), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            label = self.ds_track.short_repr
            convas = plot_utils.draw_label(convas, label, loc.tl.astype('int32'), label_color, color, 2)
        return convas

class DeepSORTTracker(DetectionBasedObjectTracker):
    def __init__(self, detector: ObjectDetector, domain: Box, tracker_conf: OmegaConf) -> None:
        super().__init__()

        self.__detector = detector
        self.det_dict = tracker_conf.get('det_mapping', DEFAULT_DETECTIION_MAPPING)

        model_file = tracker_conf.get('model_file', 'models/deepsort/model640.pt')
        model_file = Path(model_file).resolve()
        if not model_file.exists():
            if model_file.name == 'model640.pt':
                dna.utils.gdown_file('https://drive.google.com/uc?id=160jJWtgIhyhHIBpgNOkAT52uvvtOYGly', model_file)
            else:
                raise ValueError(f'Cannot find DeepSORT reid model: {model_file}')
        wt_path = Path(model_file)

        detection_threshold = tracker_conf.get('detection_threshold', DEFAULT_DETECTION_THRESHOLD)
        detection_min_size = Size2d.from_expr(tracker_conf.get('detection_min_size', DEFAULT_DETECTION_MIN_SIZE))
        detection_max_size = Size2d.from_expr(tracker_conf.get('detection_max_size', DEFAULT_DETECTION_MAX_SIZE))

        iou_dist_threshold_tight = tracker_conf.get('iou_dist_threshold_tight', DEFAULT_IOU_DIST_THRESHOLD_TIGHT)
        iou_dist_threshold = tracker_conf.get('iou_dist_threshold', DEFAULT_IOU_DIST_THRESHOLD)
        iou_dist_threshold_loose = tracker_conf.get('iou_dist_threshold_loose', DEFAULT_IOU_DIST_THRESHOLD_LOOSE)
        iou_dist_threshold_gate = tracker_conf.get('iou_dist_threshold_gate', DEFAULT_IOU_DIST_THRESHOLD_GATE)

        metric_tight_threshold = tracker_conf.get('metric_tight_threshold', DEFAULT_METRIC_TIGHT_THRESHOLD)
        metric_threshold = tracker_conf.get('metric_threshold', DEFAULT_METRIC_THRESHOLD)
        metric_gate_distance = tracker_conf.get('metric_gate_distance', DEFAULT_METRIC_GATE_DISTANCE)
        metric_gate_box_distance = tracker_conf.get('metric_gate_box_distance', DEFAULT_METRIC_GATE_BOX_DISTANCE)
        metric_min_detection_size = tracker_conf.get('metric_min_detection_size', DEFAULT_METRIC_MIN_DETECTION_SIZE)
        max_feature_count = tracker_conf.get('max_feature_count', DEFAULT_MAX_FEATURE_COUNT)

        n_init = int(tracker_conf.get('n_init', DEFAULT_N_INIT))
        max_age = int(tracker_conf.get('max_age', DEFAULT_MAX_AGE))
        overlap_supress_ratio = tracker_conf.get('overlap_supress_ratio', DEFAULT_OVERLAP_SUPRESS_RATIO)
        min_new_track_size = Size2d.from_expr(tracker_conf.get('min_new_track_size', DEFAULT_MIN_NEW_TRACK_SIZE))

        if blind_zones := tracker_conf.get("blind_zones", DEFAULT_BLIND_ZONES):
            blind_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in blind_zones]
        if exit_zones := tracker_conf.get("exit_zones", DEFAULT_EXIT_ZONES):
            exit_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in exit_zones]
        if stable_zones := tracker_conf.get("stable_zones", DEFAULT_STABLE_ZONES):
            stable_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in stable_zones]

        self.params = DNASORTParams(detection_threshold=detection_threshold,
                                    detection_min_size=detection_min_size,
                                    detection_max_size=detection_max_size,
                                    iou_dist_threshold_tight=iou_dist_threshold_tight,
                                    iou_dist_threshold=iou_dist_threshold,
                                    iou_dist_threshold_loose=iou_dist_threshold_loose,
                                    iou_dist_threshold_gate=iou_dist_threshold_gate,
                                    metric_threshold=metric_tight_threshold,
                                    metric_threshold_loose=metric_threshold,
                                    metric_gate_distance=metric_gate_distance,
                                    metric_gate_box_distance=metric_gate_box_distance,
                                    metric_min_detection_size=metric_min_detection_size,
                                    max_feature_count=max_feature_count,
                                    n_init=n_init,
                                    max_age=max_age,
                                    overlap_supress_ratio=overlap_supress_ratio,
                                    min_new_track_size=min_new_track_size,
                                    blind_zones=blind_zones,
                                    exit_zones=exit_zones,
                                    stable_zones=stable_zones)
        self.deepsort = deepsort_rbc(domain = domain,
                                    detection_threshold=self.detection_threshold,
                                    wt_path=wt_path,
                                    params=self.params)
        self.__last_frame_detections = []
        
    @property
    def detector(self) -> ObjectDetector:
        return self.__detector
        
    @property
    def detection_threshold(self) -> float:
        return self.params.detection_threshold

    def last_frame_detections(self) -> List[Detection]:
        return self.__last_frame_detections

    def track(self, frame: Frame) -> List[IDNATrack]:
        # detector를 통해 match 대상 detection들을 얻는다.
        dets:List[Detection] = self.detector.detect(frame)

        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        if self.det_dict:
            new_dets = []
            for det in dets:
                if label := self.det_dict.get(det.label, None):
                    new_dets.append(Detection(det.bbox, label, det.score))
            dets = new_dets

        # 일정 점수 이하의 detection들과 blind zone에 포함된 detection들은 무시한다.
        def is_valid_detection(det:Detection):
            return dna.utils.find_any_centroid_cover(det.bbox, self.params.blind_zones) < 0
        detections = [det for det in dets if is_valid_detection(det)]

        self.__last_frame_detections = detections
        # kwlee
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"{frame.index}: ------------------------------------------------------")
        tracker, deleted_tracks = self.deepsort.run_deep_sort(frame, detections)

        active_tracks = [DeepSORTTrack(ds_track, frame.index, frame.ts) for ds_track in tracker.tracks]
        deleted_tracks = [DeepSORTTrack(ds_track, frame.index, frame.ts) for ds_track in deleted_tracks]
        return active_tracks + deleted_tracks