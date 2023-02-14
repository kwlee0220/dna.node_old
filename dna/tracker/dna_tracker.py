
from typing import List, Union, Tuple, Any
from pathlib import Path
import sys
from enum import Enum

import numpy as np
import cv2
from omegaconf.omegaconf import OmegaConf
import shapely.geometry as geometry

from dna import Frame, Image, BGR, color, plot_utils

FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0])
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

import dna
from dna import Box, Size2d, Point
from dna.detect import ObjectDetector, Detection
from dna.tracker import ObjectTrack, TrackState, ObjectTracker, utils
from .dna_track_params import load_track_params
from .feature_extractor import FeatureExtractor
from .dna_track import DNATrack
from .tracker import Tracker

import logging
LOGGER = logging.getLogger('dna.tracker')


class DNATracker(ObjectTracker):
    def __init__(self, detector:ObjectDetector, tracker_conf:OmegaConf) -> None:
        super().__init__()

        self.detector = detector

        model_file = tracker_conf.get('model_file', 'models/deepsort/model640.pt')
        model_file = Path(model_file).resolve()
        if not model_file.exists():
            if model_file.name == 'model640.pt':
                dna.utils.gdown_file('https://drive.google.com/uc?id=160jJWtgIhyhHIBpgNOkAT52uvvtOYGly', model_file)
            else:
                raise ValueError(f'Cannot find DeepSORT reid model: {model_file}')
        wt_path = Path(model_file)
        
        self.params = load_track_params(tracker_conf)
  
        #loading this encoder is slow, should be done only once.
        self.feature_extractor = FeatureExtractor(wt_path, LOGGER)
        self.tracker = Tracker(self.params, LOGGER)

    @property
    def tracks(self) -> List[DNATrack]:
        return self.tracker.tracks

    def track(self, frame: Frame) -> List[DNATrack]:
        dna.DEBUG_FRAME_INDEX = frame.index

        # detector를 통해 match 대상 detection들을 얻는다.
        detections = self.detector.detect(frame)

        # 불필요한 detection들을 제거하고, 영상에서 각 detection별로 feature를 추출하여 부여한다.
        detections = self._prepare_detections(frame.image, detections)
        if dna.DEBUG_SHOW_IMAGE:
            self.draw_detections(frame.image.copy(), 'detections', detections)

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"{frame.index}: ------------------------------------------------------")
        session, deleted_tracks = self.tracker.track(frame, detections)
        
        return self.tracker.tracks + deleted_tracks

    def _prepare_detections(self, image:Image, detections:List[Detection]) -> List[Detection]:
        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        detections = [det for det in detections if det.label in self.params.detection_classes]

        # Detection box 크기에 따라 invalid한 detection들을 제거한다.
        # 일정 점수 이하의 detection들과 blind zone에 포함된 detection들은 무시한다.
        def is_valid_detection(det:Detection) -> bool:
            if not self.params.is_valid_size(det.bbox.size()):
                return False
            if dna.utils.find_any_centroid_cover(det.bbox, self.params.blind_zones) >= 0:
                return False
            return True
        detections = [det for det in detections if is_valid_detection(det)]
            
        # if dna.DEBUG_SHOW_IMAGE:
        #     self.draw_detections(image.copy(), 'detections', detections)

        # Detection끼리 너무 많이 겹치는 경우 low-score detection을 제거한다.
        def supress_overlaps(detections:List[Detection]) -> List[Detection]:
            remains = sorted(detections.copy(), key=lambda d: d.score, reverse=True)
            supresseds = []
            while remains:
                head = remains.pop(0)
                supresseds.append(head)
                remains = [det for det in remains if head.bbox.iou(det.bbox) < self.params.max_nms_score]
                pass
            return supresseds
        detections = supress_overlaps(detections)
        
        for det, feature in zip(detections, self.feature_extractor.extract(image, detections)):
            det.feature = feature

        return detections

    def draw_detections(self, convas:Image, title:str, detections:List[Detection], line_thickness=1):
        for idx, det in enumerate(detections):
            if det.score < self.params.detection_threshold:
                det.draw(convas, color.RED, label=str(idx), label_color=color.WHITE,
                        label_tl=Point.from_np(det.bbox.br.astype(int)), line_thickness=line_thickness)
        for idx, det in enumerate(detections):
            if det.score >= self.params.detection_threshold:
                det.draw(convas, color.BLUE, label=str(idx), label_color=color.WHITE,
                        label_tl=Point.from_np(det.bbox.br.astype(int)), line_thickness=line_thickness)
        cv2.imshow(title, convas)
        cv2.waitKey(1)
    