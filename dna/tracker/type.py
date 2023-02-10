
from typing import List, Tuple

import shapely.geometry as geometry

from dna import Box, Size2d
from dna.detect import Detection

from collections import namedtuple
IouDistThreshold = namedtuple('IouDistThreshold', 'iou,distance')


from dataclasses import dataclass, field
@dataclass(frozen=True, eq=True)    # slots=True
class DNASORTParams:
    detection_threshold: float
    detection_min_size: Size2d
    detection_max_size: Size2d

    iou_dist_threshold_tight: IouDistThreshold
    iou_dist_threshold: IouDistThreshold
    iou_dist_threshold_loose: IouDistThreshold
    iou_dist_threshold_gate: IouDistThreshold

    metric_threshold: float
    metric_threshold_loose: float
    metric_gate_distance: float
    metric_gate_box_distance: float
    metric_min_detection_size: Size2d
    max_feature_count: int

    n_init: int
    min_new_track_size: Size2d
    max_age: int
    overlap_supress_ratio: float

    blind_zones: List[geometry.Polygon]
    exit_zones: List[geometry.Polygon]
    stable_zones: List[geometry.Polygon]
    
    def is_valid_detection(self, det:Detection) -> bool:
        size = det.bbox.size()
        return self.detection_min_size.width <= size.width <= self.detection_max_size.width \
            and self.detection_min_size.height <= size.height <= self.detection_max_size.height

    def is_strong_detection(self, det:Detection) -> bool:
        return det.score >= self.detection_threshold
    
    def is_large_detection_for_metric(self, det:Detection) -> bool:
        return det.bbox.size().width >= self.metric_min_detection_size.width \
                and det.bbox.size().height >= self.metric_min_detection_size.height