
from typing import List, Union, Tuple, Any
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
DEEPSORT_DIR = str(FILE.parents[0])
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

import dna
from dna import Box, Size2d, Point
from dna.detect import ObjectDetector, Detection
from dna.tracker import ObjectTrack, TrackState, ObjectTracker, IouDistThreshold, utils
from .feature_extractor import FeatureExtractor
from .dna_track import DNATrack
from .tracker import Tracker

import logging
LOGGER = logging.getLogger('dna.tracker.dnasort')

DEFAULT_DETECTIION_CLASSES = ['car', 'bus', 'truck']
DEFAULT_DETECTION_THRESHOLD = 0.35
DEFAULT_DETECTION_MIN_SIZE = Size2d(20, 15)
DEFAULT_DETECTION_MAX_SIZE = Size2d(768, 768)

DEFAULT_IOU_DIST_THRESHOLD = IouDistThreshold(0.80, 70)
DEFAULT_IOU_DIST_THRESHOLD_LOOSE = IouDistThreshold(0.85, 90)

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


class DNATracker(ObjectTracker):
    def __init__(self, detector: ObjectDetector, domain: Box, tracker_conf: OmegaConf) -> None:
        super().__init__()

        self.__detector = detector

        model_file = tracker_conf.get('model_file', 'models/deepsort/model640.pt')
        model_file = Path(model_file).resolve()
        if not model_file.exists():
            if model_file.name == 'model640.pt':
                dna.utils.gdown_file('https://drive.google.com/uc?id=160jJWtgIhyhHIBpgNOkAT52uvvtOYGly', model_file)
            else:
                raise ValueError(f'Cannot find DeepSORT reid model: {model_file}')
        wt_path = Path(model_file)
        
        self.params = self._load_track_params(tracker_conf)
  
        #loading this encoder is slow, should be done only once.
        self.feature_extractor = FeatureExtractor(wt_path, LOGGER)
        self.tracker = Tracker(domain, self.params)
        self.__last_frame_detections = []
        
    @property
    def detector(self) -> ObjectDetector:
        return self.__detector

    @property
    def tracks(self) -> List[DNATrack]:
        return self.tracker.tracks
        
    @property
    def detection_threshold(self) -> float:
        return self.params.detection_threshold

    def last_frame_detections(self) -> List[Detection]:
        return self.__last_frame_detections

    def track(self, frame: Frame) -> List[ObjectTrack]:
        # detector를 통해 match 대상 detection들을 얻는다.
        detections:List[Detection] = self.detector.detect(frame)

        # 불필요한 detection들을 제거하고, 영상에서 각 detection별로 feature를 추출하여 부여한다.
        detections = self._prepare_detections(frame.image, detections)

        self.__last_frame_detections = detections
        # kwlee
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"{frame.index}: ------------------------------------------------------")
        session, deleted_tracks = self.tracker.track(frame, detections)
        
        return self.tracker.tracks + deleted_tracks
        
    def _load_track_params(self, track_conf:OmegaConf) -> DNASORTParams:
        detection_classes = set(track_conf.get('detection_classes', DEFAULT_DETECTIION_CLASSES))
        detection_threshold = track_conf.get('detection_threshold', DEFAULT_DETECTION_THRESHOLD)
        detection_min_size = Size2d.from_expr(track_conf.get('detection_min_size', DEFAULT_DETECTION_MIN_SIZE))
        detection_max_size = Size2d.from_expr(track_conf.get('detection_max_size', DEFAULT_DETECTION_MAX_SIZE))

        iou_dist_threshold = track_conf.get('iou_dist_threshold', DEFAULT_IOU_DIST_THRESHOLD)
        iou_dist_threshold_loose = track_conf.get('iou_dist_threshold_loose', DEFAULT_IOU_DIST_THRESHOLD_LOOSE)

        metric_tight_threshold = track_conf.get('metric_tight_threshold', DEFAULT_METRIC_TIGHT_THRESHOLD)
        metric_threshold = track_conf.get('metric_threshold', DEFAULT_METRIC_THRESHOLD)
        metric_gate_distance = track_conf.get('metric_gate_distance', DEFAULT_METRIC_GATE_DISTANCE)
        metric_gate_box_distance = track_conf.get('metric_gate_box_distance', DEFAULT_METRIC_GATE_BOX_DISTANCE)
        metric_min_detection_size = track_conf.get('metric_min_detection_size', DEFAULT_METRIC_MIN_DETECTION_SIZE)
        max_feature_count = track_conf.get('max_feature_count', DEFAULT_MAX_FEATURE_COUNT)

        n_init = int(track_conf.get('n_init', DEFAULT_N_INIT))
        max_age = int(track_conf.get('max_age', DEFAULT_MAX_AGE))
        overlap_supress_ratio = track_conf.get('overlap_supress_ratio', DEFAULT_OVERLAP_SUPRESS_RATIO)
        min_new_track_size = Size2d.from_expr(track_conf.get('min_new_track_size', DEFAULT_MIN_NEW_TRACK_SIZE))

        if blind_zones := track_conf.get("blind_zones", DEFAULT_BLIND_ZONES):
            blind_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in blind_zones]
        if exit_zones := track_conf.get("exit_zones", DEFAULT_EXIT_ZONES):
            exit_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in exit_zones]
        if stable_zones := track_conf.get("stable_zones", DEFAULT_STABLE_ZONES):
            stable_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in stable_zones]

        return DNASORTParams(detection_classes=detection_classes,
                            detection_threshold=detection_threshold,
                            detection_min_size=detection_min_size,
                            detection_max_size=detection_max_size,
                            
                            iou_dist_threshold=iou_dist_threshold,
                            iou_dist_threshold_loose=iou_dist_threshold_loose,
                            
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

    def _prepare_detections(self, image:Image, detections:List[Detection]) -> List[Detection]:
        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        detections = [det for det in detections if det.label in self.params.detection_classes]

        # Detection box 크기에 따라 invalid한 detection들을 제거한다.
        # 일정 점수 이하의 detection들과 blind zone에 포함된 detection들은 무시한다.
        def is_valid_detection(det:Detection) -> bool:
            size = det.bbox.size()
            if size.width < self.params.detection_min_size.width or size.width > self.params.detection_max_size.width \
                or size.height < self.params.detection_min_size.height or size.height > self.params.detection_max_size.height:
                return False
            if dna.utils.find_any_centroid_cover(det.bbox, self.params.blind_zones) >= 0:
                return False
            return True
        detections = [det for det in detections if is_valid_detection(det)]
            
        if dna.DEBUG_SHOW_IMAGE:
            self.draw_detections(image.copy(), 'detections', detections)

        # Detection끼리 너무 많이 겹치는 경우 low-score detection을 제거한다.
        def supress_overlaps(detections:List[Detection]) -> List[Detection]:
            remains = sorted(detections.copy(), key=lambda d: d.score, reverse=True)
            supresseds = []
            while remains:
                head = remains.pop(0)
                supresseds.append(head)
                remains = [det for det in remains if head.bbox.iou(det.bbox) < 0.8]
                pass
            return supresseds
        detections = supress_overlaps(detections)
        
        for det, feature in zip(detections, self.feature_extractor.extract(image, detections)):
            det.feature = feature
            
        if dna.DEBUG_SHOW_IMAGE:
            self.draw_detections(image.copy(), 'detections', detections)

        return detections

    def draw_detections(self, convas:Image, title:str, detections:List[Detection], line_thickness=1):
        for idx, det in enumerate(detections):
            if det.score < self.params.detection_threshold:
                det.draw(convas, color.RED, label=str(idx), label_color=color.WHITE,
                        label_tl=Point.from_np(det.bbox.br.astype(int)), line_thickness=line_thickness)
                # convas = plot_utils.draw_label(convas, str(idx), det.bbox.br.astype(int), color.WHITE, color.RED, 1)
                # convas = det.bbox.draw(convas, color.RED, line_thickness=line_thickness) 
        for idx, det in enumerate(detections):
            if det.score >= self.params.detection_threshold:
                det.draw(convas, color.BLUE, label=str(idx), label_color=color.WHITE,
                        label_tl=Point.from_np(det.bbox.br.astype(int)), line_thickness=line_thickness)
                # convas = plot_utils.draw_label(convas, str(idx), det.bbox.br.astype(int), color.WHITE, color.BLUE, 1)
                # convas = det.bbox.draw(convas, color.BLUE, line_thickness=line_thickness)
        cv2.imshow(title, convas)
        cv2.waitKey(1)
        