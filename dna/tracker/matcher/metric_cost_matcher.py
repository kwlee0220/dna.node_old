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
from .base import Matcher, chain, MatchingSession, matches_str, INVALID_METRIC_DISTANCE
from .hungarian_matcher import HungarianMatcher
from dna.tracker.dna_track import DNATrack


class MetricCostMatcher(Matcher):
    def __init__(self, tracks:List[DNATrack], detections:List[Detection],
                 params:DNATrackParams,
                 metric_cost:np.ndarray, dist_cost:np.ndarray,
                 logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.metric_cost = metric_cost
        self.dist_cost = dist_cost
        self.logger = logger
       
    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        session = MatchingSession(self.tracks, self.detections, self.params, track_idxes, det_idxes)

        metric_matcher = HungarianMatcher(self.metric_cost, threshold_name="metric",
                                            threshold=self.params.metric_threshold,
                                            invalid_value=INVALID_METRIC_DISTANCE)

        #####################################################################################################
        ################ Hot track에 한정해서 강한 threshold를 사용해서  matching 실시
        #####################################################################################################
        if session.unmatched_hot_track_idxes and session.unmatched_metric_det_idxes:
            matches0 = metric_matcher.match(session.unmatched_hot_track_idxes, session.unmatched_metric_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"hot, strong, {metric_matcher}]: {matches_str(self.tracks, matches0)}")

        #####################################################################################################
        ################ Tentative track에 한정해서 강한 threshold를 사용해서  matching 실시
        #####################################################################################################
        if session.unmatched_tentative_track_idxes and session.unmatched_metric_det_idxes:
            matches0 = metric_matcher.match(session.unmatched_tentative_track_idxes, session.unmatched_metric_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"tentative, strong, {metric_matcher}]: {matches_str(self.tracks, matches0)}")

        #####################################################################################################
        ################ 전체 track에 대해 matching 실시
        #####################################################################################################
        if session.unmatched_track_idxes and session.unmatched_metric_det_idxes:
            metric_matcher_loose = HungarianMatcher(self.metric_cost, threshold_name="metric",
                                                    threshold=self.params.metric_threshold_loose,
                                                    invalid_value=INVALID_METRIC_DISTANCE)
            matches0 = metric_matcher_loose.match(session.unmatched_track_idxes, session.unmatched_metric_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"all, strong, {metric_matcher_loose}): {matches_str(self.tracks, matches0)}")

        return session.matches