
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
DNA_TRACK_DIR = str(FILE.parents[0])
if not DNA_TRACK_DIR in sys.path:
    sys.path.append(DNA_TRACK_DIR)

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
        
        self.shrinked_rois = []
        self.roi_shifts = []
        for roi in self.params.detection_rois:
            shrinked = Box(roi.tlbr + np.array([5, 5, -5, -5]))
            if not shrinked.is_valid():
                raise ValueError(f'too small roi: {roi}')
            self.shrinked_rois.append(shrinked)
            self.roi_shifts.append(Size2d.from_np(roi.tl))

    @property
    def tracks(self) -> List[DNATrack]:
        return self.tracker.tracks

    def track(self, frame: Frame) -> List[DNATrack]:
        dna.DEBUG_FRAME_INDEX = frame.index

        detections_list = self.detector.detect_images([frame] + self.crop_rois(frame))

        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        filterds = []
        for dets in detections_list:
            filterds.append([det for det in dets if det.label in self.params.detection_classes])
        detections_list = filterds

        # Detection box 크기에 따라 invalid한 detection들을 제거한다.
        filterds = []
        for dets in detections_list:
            filterds.append([det for det in dets if self.params.is_valid_size(det.bbox.size())])
        detections_list = filterds

        mergeds = self.merge(detections_list)

        # blind zone에 포함된 detection들은 무시한다.
        detections = [det for det in mergeds if dna.utils.find_any_centroid_cover(det.bbox, self.params.blind_zones) < 0]

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
        
        # Filtering을 마친 detection에 대해서는 영상 내의 해당 영역에서 feature를 추출하여 부여한다.
        for det, feature in zip(detections, self.feature_extractor.extract(frame.image, detections)):
            det.feature = feature

        if dna.DEBUG_SHOW_IMAGE:
            convas = frame.image.copy()
            for roi in self.params.detection_rois:
                convas = roi.draw(convas, color.YELLOW, line_thickness=1)
            self.draw_detections(convas, 'detections', detections)

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(f"{frame.index}: ------------------------------------------------------")
        session, deleted_tracks = self.tracker.track(frame, detections)
        
        return self.tracker.tracks + deleted_tracks

    def crop_rois(self, frame:Frame) -> List[Frame]:
        return [Frame(roi.crop(frame.image), frame.index, frame.ts) for roi in self.params.detection_rois]

    def merge(self, detections_list:List[Detection]) -> List[Detection]:
        mergeds = []

        cropped_detections_list = detections_list[1:]
        for roi, shift, dets in zip(self.shrinked_rois, self.roi_shifts, cropped_detections_list):
            for det in dets:
                shifted_box = det.bbox.translate(shift)
                if roi.contains(shifted_box):
                    mergeds.append(Detection(shifted_box, det.label, det.score))

        for det in detections_list[0]:
            if all(not roi.contains(det.bbox) for roi in self.params.detection_rois):
                mergeds.append(det)

        return mergeds

    # def _prepare_detections(self, image:Image, detections:List[Detection]) -> List[Detection]:
    #     # 검출 물체 중 관련있는 label의 detection만 사용한다.
    #     detections = [det for det in detections if det.label in self.params.detection_classes]

    #     # Detection box 크기에 따라 invalid한 detection들을 제거한다.
    #     # 일정 점수 이하의 detection들과 blind zone에 포함된 detection들은 무시한다.
    #     def is_valid_detection(det:Detection) -> bool:
    #         if not self.params.is_valid_size(det.bbox.size()):
    #             return False
    #         if dna.utils.find_any_centroid_cover(det.bbox, self.params.blind_zones) >= 0:
    #             return False
    #         return True
    #     detections = [det for det in detections if is_valid_detection(det)]
            
    #     # if dna.DEBUG_SHOW_IMAGE:
    #     #     self.draw_detections(image.copy(), 'detections', detections)

    #     # Detection끼리 너무 많이 겹치는 경우 low-score detection을 제거한다.
    #     def supress_overlaps(detections:List[Detection]) -> List[Detection]:
    #         remains = sorted(detections.copy(), key=lambda d: d.score, reverse=True)
    #         supresseds = []
    #         while remains:
    #             head = remains.pop(0)
    #             supresseds.append(head)
    #             remains = [det for det in remains if head.bbox.iou(det.bbox) < self.params.max_nms_score]
    #             pass
    #         return supresseds
    #     detections = supress_overlaps(detections)
        
    #     for det, feature in zip(detections, self.feature_extractor.extract(image, detections)):
    #         det.feature = feature

    #     return detections

    def draw_detections(self, convas:Image, title:str, detections:List[Detection], line_thickness=1):
        for roi, shrinked in zip(self.params.detection_rois, self.shrinked_rois):
            convas = roi.draw(convas, color.YELLOW, line_thickness=1)
            # convas = shrinked.draw(convas, color.WHITE, line_thickness=1)

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
    