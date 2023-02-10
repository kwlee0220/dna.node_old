from __future__ import annotations
from typing import Tuple, Iterable, List, Optional

import logging
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

import dna
from dna.detect import Detection
from dna.tracker import utils, DNASORTParams
from .base import Matcher, chain, MatchingSession, matches_str
from .hungarian_matcher import HungarianMatcher
from dna.tracker.dna_deepsort.deepsort.track import Track

from .nn_matching import _nn_cosine_distance   


class MetricCostMatcher(Matcher):
    def __init__(self, tracks:List[Track], detections:List[Detection],
                 params:DNASORTParams,
                 metric_cost:np.ndarray, dist_cost:np.ndarray,
                 logger:Optional[logging.Logger]=None) -> None:
        self.tracks = tracks
        self.detections = detections
        self.params = params
        self.metric_cost = metric_cost
        self.dist_cost = dist_cost
        self.metric_distance_func = _nn_cosine_distance
        self.logger = logger
       
    def match(self, track_idxes:Iterable[int], det_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        session = MatchingSession(self.tracks, self.detections, self.params, track_idxes, det_idxes)

        hung_matcher = HungarianMatcher(self.metric_cost, None, 1)
        METRIC_DISTANCE = self.params.metric_threshold
        METRIC_DISTANCE_LOOSE = self.params.metric_threshold_loose

        #####################################################################################################
        ################ Hot track에 한정해서 강한 threshold를 사용해서  matching 실시
        ################ Tentative track에 비해 2배 이상 먼거리를 갖는 경우에는 matching을 하지 않도록 함.
        #####################################################################################################
        if session.unmatched_hot_track_idxes and session.unmatched_metric_det_idxes:
            matches0 = hung_matcher.match_threshold(METRIC_DISTANCE, session.unmatched_hot_track_idxes,
                                                    session.unmatched_metric_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"hot, strong, metric[{METRIC_DISTANCE}]: {matches_str(self.tracks, matches0)}")

        #####################################################################################################
        ################ Tentative track에 한정해서 강한 threshold를 사용해서  matching 실시
        #####################################################################################################
        if session.unmatched_tentative_track_idxes and session.unmatched_metric_det_idxes:
            matches0 = hung_matcher.match_threshold(METRIC_DISTANCE, session.unmatched_tentative_track_idxes,
                                                    session.unmatched_metric_det_idxes)
            if matches0:
                session.update(matches0)
                if dna.DEBUG_PRINT_COST:
                    self.logger.debug(f"tentative, strong, metric[{METRIC_DISTANCE}]: {matches_str(self.tracks, matches0)}")

        #####################################################################################################
        ################ 전체 track에 대해 matching 실시
        #####################################################################################################
        if session.unmatched_track_idxes and session.unmatched_metric_det_idxes:
            matches0 = hung_matcher.match_threshold(METRIC_DISTANCE_LOOSE, session.unmatched_track_idxes,
                                                    session.unmatched_metric_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"all, strong, metric_loose[{METRIC_DISTANCE_LOOSE}]): {matches_str(self.tracks, matches0)}")

        return session.matches