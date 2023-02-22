from __future__ import annotations
from typing import Tuple, Iterable, List, Optional

import logging
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

import dna
from dna.detect import Detection
from dna.tracker import utils
from ..dna_track_params import DNATrackParams
from dna.tracker.matcher import Matcher, chain, MatchingSession, matches_str, INVALID_METRIC_DISTANCE
from dna.tracker.matcher.cost_matrices import build_metric_cost, gate_metric_cost
from .hungarian_matcher import HungarianMatcher
from dna.tracker.dna_track import DNATrack


class MetricCostMatcher(Matcher):
    def __init__(self, tracks:List[DNATrack], detections:List[Detection],
                 params:DNATrackParams, metric_cost:np.ndarray,
                 logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.metric_cost = metric_cost
        self.logger = logger
                
        self.metric_matcher = HungarianMatcher(metric_cost, threshold_name="metric",
                                                threshold=self.params.metric_threshold,
                                                invalid_value=INVALID_METRIC_DISTANCE)
       
    def match(self, track_idxes:List[int], det_idxes:List[int]) -> List[Tuple[int,int]]:
        matches0 = self.metric_matcher.match(track_idxes, det_idxes)
        return matches0
    
    def find_owner_stable_track_idx(self, session:MatchingSession, det_idx:int):
        zid = self.params.find_stable_zone(self.detections[det_idx].bbox)
        if zid < 0:
            return None
        
        owner_match = session.find_match_by_det(det_idx)
        if not owner_match:
            return None
        
        if (hz := self.tracks[owner_match[0]].home_zone) and hz == zid:
            return owner_match[0]
        else:
            return None
        
            