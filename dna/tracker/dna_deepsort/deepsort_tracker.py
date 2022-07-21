from datetime import datetime
from typing import List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys
from enum import Enum

import numpy as np
import cv2
from omegaconf.omegaconf import OmegaConf
import logging

from dna import Frame

FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0] / 'deepsort')
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

import dna
from dna import Box, Size2d, utils, gdown_file
from dna.detect import ObjectDetector, Detection
from ..tracker import Track, TrackState, DetectionBasedObjectTracker
from .deepsort.deepsort import deepsort_rbc
from .deepsort.track import Track as DSTrack
from .deepsort.track import TrackState as DSTrackState

DEFAULT_MIN_DETECTION_SCORE = 0
DEFAULT_METRIC_THRESHOLD = 0.55
DEFAULT_MAX_IOU_DISTANCE = 0.85
DEFAULT_MAX_AGE = 10
DEFAULT_N_INIT = 3
DEFAULT_MAX_OVERLAP_RATIO=0.75
DEFAULT_MIN_SIZE=[30, 20]
DEFAULT_DET_MAPPING = {'car':'car', 'bus':'car', 'truck':'car'}

@dataclass(frozen=True, eq=True)    # slots=True
class DeepSORTParams:
    metric_threshold: float
    max_iou_distance: float
    n_init: int
    max_age: int
    min_size: Size2d
    max_overlap_ratio: float
    blind_zones: List[Box]
    dim_zones: List[Box]

class DeepSORTTracker(DetectionBasedObjectTracker):
    def __init__(self, detector: ObjectDetector, domain: Box, tracker_conf: OmegaConf) -> None:
        super().__init__()

        self.__detector = detector
        self.det_dict = tracker_conf.get('det_mapping', DEFAULT_DET_MAPPING)
        self.min_detection_score = tracker_conf.get('min_detection_score', DEFAULT_METRIC_THRESHOLD)

        model_file = tracker_conf.get('model_file', 'models/deepsort/model640.pt')
        model_file = Path(model_file).resolve()
        if not model_file.exists():
            if model_file.name == 'model640.pt':
                gdown_file('https://drive.google.com/uc?id=160jJWtgIhyhHIBpgNOkAT52uvvtOYGly', model_file)
            else:
                raise ValueError(f'Cannot find DeepSORT reid model: {model_file}')
        wt_path = Path(model_file)

        metric_threshold = tracker_conf.get('metric_threshold', DEFAULT_METRIC_THRESHOLD)
        max_iou_distance = tracker_conf.get('max_iou_distance', DEFAULT_MAX_IOU_DISTANCE)
        max_age = int(tracker_conf.get('max_age', DEFAULT_MAX_AGE))
        n_init = int(tracker_conf.get('n_init', DEFAULT_N_INIT))
        max_overlap_ratio = tracker_conf.get('max_overlap_ratio', DEFAULT_MAX_OVERLAP_RATIO)
        min_size = Size2d.from_np(np.array(tracker_conf.get('min_size', DEFAULT_MIN_SIZE)))

        blind_zones = tracker_conf.get("blind_zones", [])
        if len(blind_zones) > 0:
            blind_zones = [Box.from_tlbr(np.array(zone, dtype=np.int32)) for zone in blind_zones]

        dim_zones = tracker_conf.get("dim_zones", [])
        if len(dim_zones) > 0:
            dim_zones = [Box.from_tlbr(np.array(zone, dtype=np.int32)) for zone in dim_zones]

        self.params = DeepSORTParams(metric_threshold=metric_threshold,
                                    max_iou_distance=max_iou_distance,
                                    max_age=max_age,
                                    n_init=n_init,
                                    max_overlap_ratio=max_overlap_ratio,
                                    min_size=min_size,
                                    blind_zones=blind_zones,
                                    dim_zones=dim_zones)
        self.deepsort = deepsort_rbc(domain = domain,
                                    wt_path=wt_path,
                                    params=self.params)
        self.__last_frame_detections = []

        # level_name = tracker_conf.get("log_level", "info").upper()
        # level = logging.getLevelName(level_name)
        # logger = get_logger("dna.track.deep_sort")
        # logger.setLevel(level)
        
    @property
    def detector(self) -> ObjectDetector:
        return self.__detector

    def last_frame_detections(self) -> List[Detection]:
        return self.__last_frame_detections

    def __replace_detection_label(self, det) -> Union[Detection,None]:
        label = self.det_dict.get(det.label, None)
        if label:
            return Detection(det.bbox, label, det.score)
        else:
            return None

    def track(self, frame: Frame) -> List[Track]:
        # detector를 통해 match 대상 detection들을 얻는다.
        dets = self.detector.detect(frame)

        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        if self.det_dict:
            new_dets = []
            for det in dets:
                label = self.det_dict.get(det.label, None)
                if label:
                    new_dets.append(Detection(det.bbox, label, det.score))
            dets = new_dets

        # 일정 점수 이하의 detection들과 blind zone에 포함된 detection들은 무시한다.
        def is_valid_detection(det):
            return det.score >= self.min_detection_score and \
                    not any(zone.contains(det.bbox) for zone in self.params.blind_zones)
        detections = [det for det in dets if is_valid_detection(det)]

        self.__last_frame_detections = detections
        bboxes, scores = self.split_boxes_scores(self.__last_frame_detections)
        tracker, deleted_tracks = self.deepsort.run_deep_sort(frame.image.astype(np.uint8), bboxes, scores)

        active_tracks = [self.to_dna_track(ds_track, frame.index, frame.ts) for ds_track in tracker.tracks]
        deleted_tracks = [self.to_dna_track(ds_track, frame.index, frame.ts) for ds_track in deleted_tracks]
        return active_tracks + deleted_tracks

    def to_dna_track(self, ds_track: DSTrack, frame_idx: int, ts:float) -> Track:
        if ds_track.state == DSTrackState.Confirmed:
            state = TrackState.Confirmed if ds_track.time_since_update <= 0 else TrackState.TemporarilyLost
        elif ds_track.state == DSTrackState.Tentative:
            state = TrackState.Tentative
        elif ds_track.state == DSTrackState.Deleted:
            state = TrackState.Deleted

        return Track(id=ds_track.track_id, state=state,
                    location=Box.from_tlbr(np.rint(ds_track.to_tlbr())),
                    frame_index=frame_idx, ts=ts)

    def split_boxes_scores(self, det_list) -> Tuple[List[Box], List[float]]:
        boxes = []
        scores = []
        for det in det_list:
            boxes.append(det.bbox)
            scores.append(det.score)
        
        return np.array(boxes), scores