# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Tuple, Set, Dict, Generator

from collections import defaultdict
import logging
import numpy as np
import numpy.typing as npt
from numpy.linalg import det
import cv2

import dna
from dna import Box, Size2d, Frame, plot_utils, color, Image, Point
from dna.detect import Detection
from dna.tracker import utils
from dna.tracker.dna_track_params import DistanceIoUThreshold
from dna.tracker.matcher import Matcher, MatchingSession, chain, matches_str, match_str, \
                                IoUDistanceCostMatcher, MetricCostMatcher, HungarianMatcher, ReciprocalCostMatcher, \
                                INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE, INVALID_METRIC_DISTANCE
from dna.tracker.matcher.cost_matrices import build_dist_cost, build_iou_cost, gate_dist_iou_cost, \
                                                build_metric_cost, gate_metric_cost
from .kalman_filter import KalmanFilter
from .dna_track_params import DNATrackParams
from .dna_track import DNATrack
from dna.node.types import TrackEvent

_EMPTY_FEATURE = np.zeros(1024)

class Tracker:
    def __init__(self, params:DNATrackParams, logger:logging.Logger):
        self.params = params
        self.kf = KalmanFilter()
        self.tracks:List[DNATrack] = []
        self._next_id = 1
        self.logger = logger

    def track(self, frame:Frame, detections: List[Detection]) -> Tuple[MatchingSession, List[DNATrack], List[TrackEvent]]:
        # Estimate the next state for each tracks using Kalman filter
        for track in self.tracks:
            track.predict(self.kf)

        # Run matching
        session = self.match(detections)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'{session}')

        if dna.DEBUG_SHOW_IMAGE:
            # display matching track and detection pairs
            self.draw_matched_detections("detections", frame.image.copy(), session.matches, detections)

        # Update locations of tracks from their matched detections.
        for track, det in session.associations:
            track.update(self.kf, frame, det)
        
        # unmatch된 track들 중에서 해당 box의 크기가 일정 범위가 넘으면 delete시킴.
        # 그렇지 않은 track은 temporarily lost된 것으로 간주함.
        for tidx in session.unmatched_track_idxes:
            track = self.tracks[tidx]
            if not track.is_deleted():
                if self.params.detection_max_size and track.location.size() > self.params.detection_max_size:
                    track.mark_deleted()
                elif track.exit_zone >= 0:
                    # 바로 이전 frame에서 'exit-zone'에 있던 detection과 match되었던 경우는 바로 delete시킴.
                    track.mark_deleted()
                else:
                    track.mark_missed(frame)
                
        for didx in session.unmatched_strong_det_idxes:
            det = detections[didx]

            # Exit 영역에 포함되는 detection들은 무시한다
            if det.exit_zone >= 0:
                continue
            
            # create a new (tentative) track for this unmatched detection
            self._initiate_track(det, frame)
            
        merged_tracks = set()
        track_event_list = []
        if self.params.stable_zones:
            merged_tracks = self.merge_fragment(session, frame, track_event_list)

        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        for track in self.tracks:
            if track not in merged_tracks:
                track_event_list.append(track.to_track_event())
        for track in deleted_tracks:
            track_event_list.append(track.to_track_event())

        return (session, deleted_tracks, track_event_list)

    def match(self, detections:List[Detection]) -> MatchingSession:
        session = MatchingSession(self.tracks, detections, self.params)
        if not(detections and self.tracks):
            # Detection 작업에서 아무런 객체를 검출하지 못한 경우.
            return session

        # 이전 track 객체와 새로 detection된 객체사이의 거리 값을 기준으로 cost matrix를 생성함.
        dist_cost = build_dist_cost(self.kf, self.tracks, detections)
        iou_cost = build_iou_cost(self.tracks, detections)
        
        dist_iou_matcher = IoUDistanceCostMatcher(self.tracks, detections, self.params,
                                                  dist_cost, iou_cost, self.logger.getChild('matcher'))
        self.match_by_motion(session, dist_iou_matcher)
        
        ###########################################################################################################
        ###  지금까지 match되지 못한 strong detection들 중에서 이미 matching된 detection들과 많이 겹치는 경우,
        ###  이미 matching된 detection과 score를 기준으로 비교하여 더 높은 score를 갖는 detection으로 재 matching 시킴.
        ###########################################################################################################
        if session.unmatched_strong_det_idxes:
            def select_overlaps(box:Box, candidates:List[int]):
                return [idx for idx in candidates if box.iou(d_boxes[idx]) >= self.params.match_overlap_score]

            d_boxes = [det.bbox for det in detections]
            for match  in session.matches:
                matched_det_box = d_boxes[match[1]]
                overlap_det_idxes = select_overlaps(matched_det_box, session.unmatched_strong_det_idxes)
                if overlap_det_idxes:
                    candidates = [d_idx for d_idx in overlap_det_idxes] + [match[1]]
                    ranks = sorted(candidates, key=lambda i: detections[i].score, reverse=True)
                    new_match = (match[0], ranks[0])
                    session.pull_out(match)
                    session.update([new_match])
                    session.remove_det_idxes(ranks[1:])
                    if self.logger.isEnabledFor(logging.DEBUG) and match[1] != ranks[0]:
                        self.logger.debug(f'rematch: {match_str(self.tracks, match)} -> {match_str(self.tracks, new_match)}')
                    if len(session.unmatched_strong_det_idxes) == 0:
                        break
                
        ###########################################################################################################
        ### 이 단계까지 오면 지난 frame까지 active하게 추적되던 track들 (hot_track_idxes, tentative_track_idxes)에
        ### 대한 motion 정보만을 통해 matching이 완료됨.
        ### 남은 track들의 경우에는 이전 몇 frame동안 추적되지 못한 track들이어서 motion 정보만으로 matching하기
        ### 어려운 것들만 존재함. 이 track들에 대한 matching을 위해서는 appearance를 사용한 matching을 시도한다.
        ### Appearance를 사용하는 경우는 추적의 안정성을 위해 다음의 조건을 만족하는 detection에 대해서만 matching을 시도함.
        ###     - strong (high-scored) detection
        ###     - Detection box의 크기가 일정 이상이어서 추출된 metric 값이 신뢰할 수 있는 detection
        ###     - Exit-zone에 존재하지 않는 detection
        ###########################################################################################################
        self.match_by_metric(session, detections, dist_cost)

        # 아직 match되지 못한 track이 존재하면, strong detection들과 distance & IoU에 기반한
        # Hungarian 방식으로 최종 matching을 시도함.
        self.match_by_hungarian(session, iou_cost, dist_cost)
                        
        self.revise_matches(dist_iou_matcher.matcher, session, detections)

        return session

    def _initiate_track(self, detection: Detection, frame:Frame) -> None:
        mean, covariance = self.kf.initiate(detection.bbox.to_xyah())
        track = DNATrack(mean, covariance, self._next_id, frame.index, frame.ts,
                            self.params, detection)
        self.tracks.append(track)
        self._next_id += 1

    def match_by_motion(self, session:MatchingSession, dist_iou_matcher:IoUDistanceCostMatcher) -> None:
        matches0 = dist_iou_matcher.match(session.unmatched_track_idxes, session.unmatched_det_idxes)
        if matches0:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"(motion) hot+tent, all: {matches_str(self.tracks, matches0)}")
            session.update(matches0)

    def match_by_metric(self, session:MatchingSession, detections:List[Detection], dist_cost:np.array) -> None:
        unmatched_track_idxes = session.unmatched_track_idxes
        unmatched_metric_det_idxes = session.unmatched_metric_det_idxes
        if unmatched_track_idxes and unmatched_metric_det_idxes:
            metric_cost = build_metric_cost(self.tracks, detections, unmatched_track_idxes, unmatched_metric_det_idxes)
            gated_metric_cost = gate_metric_cost(metric_cost, dist_cost, self.params.metric_gate_distance)
            metric_matcher = MetricCostMatcher(gated_metric_cost, self.params.metric_threshold, self.logger)
            matches0 = metric_matcher.match(unmatched_track_idxes, unmatched_metric_det_idxes)
            if matches0:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"(metric) all, metric: {matches_str(self.tracks, matches0)}")     
                session.update(matches0)

    def match_by_hungarian(self, session:MatchingSession, iou_cost:np.array, dist_cost:np.array) -> None:
        unmatched_track_idxes = session.unmatched_track_idxes
        unmatched_strong_det_idxes = session.unmatched_strong_det_idxes
        if unmatched_track_idxes and unmatched_strong_det_idxes:
            iou_last_matcher = HungarianMatcher(iou_cost, self.params.iou_dist_threshold_loose.iou, INVALID_IOU_DISTANCE)
            dist_last_matcher = HungarianMatcher(dist_cost, self.params.iou_dist_threshold_loose.distance, INVALID_DIST_DISTANCE)
            last_resort_matcher = chain(iou_last_matcher, dist_last_matcher)

            if unmatched_strong_det_idxes:
                matches0 = last_resort_matcher.match(unmatched_track_idxes, unmatched_strong_det_idxes)
                if matches0:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"5. (motion) all, strong, last_resort[{last_resort_matcher}]: "
                                    f"{matches_str(self.tracks, matches0)}")
                    session.update(matches0)


    def revise_matches(self, matcher:Matcher, session:MatchingSession, detection: Detection) -> None:
        # tentative track과 strong detection 사이의 match들을 검색한다.
        tent_matched_strong_det_idxes = [m[1] for m in session.matches \
                                                    if self.tracks[m[0]].is_tentative() \
                                                        and self.params.is_strong_detection(detection[m[1]])]
        strong_det_idxes = session.unmatched_strong_det_idxes + tent_matched_strong_det_idxes
        if not strong_det_idxes:
            return
        
        # weak detection들과 match된 non-tentative track들을 검색함
        track_idxes = [m[0] for m in session.matches \
                                    if not self.tracks[m[0]].is_tentative() \
                                        and not self.params.is_strong_detection(detection[m[1]])]
        # Matching되지 못했던 confirmed track들을 추가한다
        track_idxes += session.unmatched_confirmed_track_idxes
        if not track_idxes:
            return
        
        matches0 = matcher.match(track_idxes, strong_det_idxes)
        if matches0:
            deprived_track_idxes = []
            for match in matcher.match(track_idxes, strong_det_idxes):
                old_weak_match = session.find_match_by_track(match[0])
                old_strong_match = session.find_match_by_det(match[1])
                if old_weak_match:
                    session.pull_out(old_weak_match)
                    deprived_track_idxes.append(old_weak_match[0])
                if old_strong_match:
                    session.pull_out(old_strong_match)
                    deprived_track_idxes.append(old_strong_match[0])
                    
                session.update([match])
                deprived_track_idxes.remove(match[0])
                if self.logger.isEnabledFor(logging.DEBUG):
                    old_weak_str = [match_str(self.tracks, old_weak_match)] if old_weak_match else []
                    old_strong_str = [match_str(self.tracks, old_strong_match)] if old_strong_match else []
                    old_str = ', '.join(old_weak_str+old_strong_str)
                    self.logger.debug(f'rematch: {old_str} -> {match_str(self.tracks, match)}')
                          
            matches1 = matcher.match(deprived_track_idxes, session.unmatched_det_idxes)
            if matches1:
                session.update(matches1)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"rematch yielding tracks, {matcher}: {matches_str(self.tracks, matches1)}")
    
    def merge_fragment(self, session:MatchingSession, frame:Frame, track_events:List[TrackEvent]) -> Set[DNATrack]:
        merged_tracks = set()
        if not (stable_home_tracks := self.build_stable_home_tracks(session)):
            return merged_tracks
        
        for zid, _ in enumerate(self.params.stable_zones):
            tl_tracks = self.find_tlost_stable_tracks(zid, session)
            sh_tracks = stable_home_tracks[zid]
            if tl_tracks and sh_tracks:
                metric_cost = self.build_metric_cost(tl_tracks, sh_tracks)

                matcher = ReciprocalCostMatcher(metric_cost, self.params.metric_threshold, name='take_over')
                matches = matcher.match(utils.all_indices(tl_tracks), utils.all_indices(sh_tracks))
                for t_idx, d_idx in matches:
                    stable_home_track = stable_home_tracks[zid][d_idx]
                    tl_tracks[t_idx].take_over(stable_home_track, self.kf, frame, self.params, track_events)
                    merged_tracks.add(tl_tracks[t_idx])
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'track-take-over: {stable_home_track.id} -> {tl_tracks[t_idx].id}: '
                                            f'count={len(stable_home_track.detections)} stable_zone[{zid}]')
        return merged_tracks
        
    def find_tlost_stable_tracks(self, zid:int, session:MatchingSession) -> List[DNATrack]:
        return [track for track in utils.get_items(self.tracks, session.unmatched_tlost_track_idxes) \
                            if track.archived_state.stable_zone == zid]

    def build_stable_home_tracks(self, session:MatchingSession) -> Dict[int,List[DNATrack]]:
        stable_home_tracks = defaultdict(list)
        for t_idx, track in enumerate(self.tracks):
            # track의 처음 생성될 때 특정 stable zone 내에 위치하였고,
            # 현재 상태가 Confirmed 또는 tentative 상태이고
            # 현 track의 현재 위치가 아직 자신이 생성된 stable zone내인 경우
            if (zid:=track.home_zone) >= 0 \
                and (track.is_confirmed() or track.is_tentative()) \
                and track.stable_zone == zid:
                    stable_home_tracks[zid].append(track)
        return stable_home_tracks

    def build_metric_cost(self, tl_tracks:List[DNATrack], sh_tracks:List[DNATrack]) -> np.ndarray:
        cost_matrix = np.ones((len(tl_tracks), len(sh_tracks)))
        for i, tl_track in enumerate(tl_tracks):
            start_index = tl_track.archived_state.frame_index
            if tl_track.features:
                for j, sh_track in enumerate(sh_tracks):
                    if sh_track.last_detection.feature is not None \
                        and sh_track.first_frame_index > start_index:
                        features = np.array([sh_track.last_detection.feature])
                        distances = utils.cosine_distance(tl_track.features, features)
                        cost_matrix[i, j] = distances.min(axis=0)
        return cost_matrix

    def draw_matched_detections(self, title:str, convas:Image, matches:List[Tuple[int,int]], detections:List[Detection]):
        # for zone in self.tracker.params.blind_zones:
        #     convas = zone.draw(convas, list(zone.exterior.coords), color.YELLOW, 1)
        for zone in self.params.exit_zones:
            convas = zone.draw(convas, color.RED, 1)
        for zone in self.params.stable_zones:
            convas = zone.draw(convas, color.BLUE, 1)

        if matches:
            for t_idx, d_idx in matches:
                label = f'{d_idx}({self.tracks[t_idx].id})'
                det = detections[d_idx]
                if self.params.is_strong_detection(det):
                    convas = plot_utils.draw_label(convas, label, Point.from_np(det.bbox.br.astype(int)), color.WHITE, color.BLUE, 1)
                    convas = det.bbox.draw(convas, color.BLUE, line_thickness=1)
                else:
                    convas = plot_utils.draw_label(convas, label, Point.from_np(det.bbox.br.astype(int)), color.WHITE, color.RED, 1)
                    convas = det.bbox.draw(convas, color.RED, line_thickness=1)
        cv2.imshow(title, convas)
        cv2.waitKey(1)