from __future__ import annotations
from typing import List, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field

from omegaconf.omegaconf import OmegaConf
import shapely.geometry as geometry

from dna import Size2d, Box
from dna.detect import Detection

from collections import namedtuple
IouDistThreshold = namedtuple('IouDistThreshold', 'iou,distance')

DEFAULT_DETECTIION_CLASSES = ['car', 'bus', 'truck']
DEFAULT_DETECTION_THRESHOLD = 0.37
DEFAULT_DETECTION_MIN_SIZE = Size2d(15, 15)
DEFAULT_DETECTION_MAX_SIZE = Size2d(768, 768)
DEFAULT_DETECTION_ROIS = []

DEFAULT_IOU_DIST_THRESHOLD = IouDistThreshold(0.80, 70)
DEFAULT_IOU_DIST_THRESHOLD_LOOSE = IouDistThreshold(0.85, 90)

DEFAULT_METRIC_TIGHT_THRESHOLD = 0.3
DEFAULT_METRIC_THRESHOLD = 0.55
DEFAULT_METRIC_GATE_DISTANCE = 500
DEFAULT_METRIC_MIN_DETECTION_SIZE = Size2d(40, 35)
DEFAULT_MAX_FEATURE_COUNT = 50

DEFAULT_N_INIT = 3
DEFAULT_MAX_AGE = 10
DEFAULT_NEW_TRACK_MIN_SIZE = [30, 20]
DEFAULT_MATCH_OVERLAP_SCORE = 0.75
DEFAULT_MAX_NMS_SCORE = 0.8

DEFAULT_BLIND_ZONES = []
DEFAULT_EXIT_ZONES = []
DEFAULT_STABLE_ZONES = []

@dataclass(frozen=True, eq=True)    # slots=True
class DNATrackParams:
    detection_classes: Set[str]
    detection_threshold: float
    detection_min_size: Size2d
    detection_max_size: Size2d
    detection_rois: List[Box]

    iou_dist_threshold: IouDistThreshold
    iou_dist_threshold_loose: IouDistThreshold

    metric_threshold: float
    metric_threshold_loose: float
    metric_gate_distance: float
    metric_min_detection_size: Size2d
    max_feature_count: int

    n_init: int
    new_track_min_size: Size2d
    max_age: int
    match_overlap_score: float
    max_nms_score: float

    blind_zones: List[geometry.Polygon]
    exit_zones: List[geometry.Polygon]
    stable_zones: List[geometry.Polygon]

    def is_valid_size(self, size:Size2d) -> bool:
        return self.detection_min_size.width <= size.width <= self.detection_max_size.width \
            and self.detection_min_size.height <= size.height <= self.detection_max_size.height

    def is_strong_detection(self, det:Detection) -> bool:
        return det.score >= self.detection_threshold
    
    def is_large_detection_for_metric(self, det:Detection) -> bool:
        return det.bbox.size().width >= self.metric_min_detection_size.width \
                and det.bbox.size().height >= self.metric_min_detection_size.height


def load_track_params(track_conf:OmegaConf) -> DNATrackParams:
    detection_classes = set(track_conf.get('detection_classes', DEFAULT_DETECTIION_CLASSES))
    detection_threshold = track_conf.get('detection_threshold', DEFAULT_DETECTION_THRESHOLD)
    detection_min_size = Size2d.from_expr(track_conf.get('detection_min_size', DEFAULT_DETECTION_MIN_SIZE))
    detection_max_size = Size2d.from_expr(track_conf.get('detection_max_size', DEFAULT_DETECTION_MAX_SIZE))
    detection_rois = [Box.from_tlbr(roi) for roi in track_conf.get('rois', DEFAULT_DETECTION_ROIS)]

    iou_dist_threshold = track_conf.get('iou_dist_threshold', DEFAULT_IOU_DIST_THRESHOLD)
    iou_dist_threshold_loose = track_conf.get('iou_dist_threshold_loose', DEFAULT_IOU_DIST_THRESHOLD_LOOSE)

    metric_tight_threshold = track_conf.get('metric_tight_threshold', DEFAULT_METRIC_TIGHT_THRESHOLD)
    metric_threshold = track_conf.get('metric_threshold', DEFAULT_METRIC_THRESHOLD)
    metric_gate_distance = track_conf.get('metric_gate_distance', DEFAULT_METRIC_GATE_DISTANCE)
    metric_min_detection_size = track_conf.get('metric_min_detection_size', DEFAULT_METRIC_MIN_DETECTION_SIZE)
    max_feature_count = track_conf.get('max_feature_count', DEFAULT_MAX_FEATURE_COUNT)

    n_init = int(track_conf.get('n_init', DEFAULT_N_INIT))
    max_age = int(track_conf.get('max_age', DEFAULT_MAX_AGE))
    max_nms_score = track_conf.get('max_nms_score', DEFAULT_MAX_NMS_SCORE)
    match_overlap_score = track_conf.get('match_overlap_score', DEFAULT_MATCH_OVERLAP_SCORE)
    new_track_min_size = Size2d.from_expr(track_conf.get('new_track_min_size', DEFAULT_NEW_TRACK_MIN_SIZE))

    if blind_zones := track_conf.get("blind_zones", DEFAULT_BLIND_ZONES):
        blind_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in blind_zones]
    if exit_zones := track_conf.get("exit_zones", DEFAULT_EXIT_ZONES):
        exit_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in exit_zones]
    if stable_zones := track_conf.get("stable_zones", DEFAULT_STABLE_ZONES):
        stable_zones = [geometry.Polygon([tuple(c) for c in zone]) for zone in stable_zones]

    return DNATrackParams(detection_classes=detection_classes,
                        detection_threshold=detection_threshold,
                        detection_min_size=detection_min_size,
                        detection_max_size=detection_max_size,
                        detection_rois=detection_rois,
                        
                        iou_dist_threshold=iou_dist_threshold,
                        iou_dist_threshold_loose=iou_dist_threshold_loose,
                        
                        metric_threshold=metric_tight_threshold,
                        metric_threshold_loose=metric_threshold,
                        metric_gate_distance=metric_gate_distance,
                        metric_min_detection_size=metric_min_detection_size,
                        max_feature_count=max_feature_count,
                        
                        n_init=n_init,
                        max_age=max_age,
                        max_nms_score=max_nms_score,
                        match_overlap_score=match_overlap_score,
                        new_track_min_size=new_track_min_size,
                        
                        blind_zones=blind_zones,
                        exit_zones=exit_zones,
                        stable_zones=stable_zones)