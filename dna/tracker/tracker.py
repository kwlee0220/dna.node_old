# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Tuple, Set

import logging
import numpy as np
import numpy.typing as npt
from numpy.linalg import det
import cv2

import dna
from dna import Box, Size2d, Frame, plot_utils, color, Image, Point
from dna.detect import Detection
from dna.tracker import utils
from dna.tracker.matcher import Matcher, MatchingSession, chain, matches_str, match_str, \
                                IoUDistanceCostMatcher, MetricCostMatcher, HungarianMatcher, \
                                INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE
from dna.tracker.matcher.cost_matrices import build_dist_iou_cost, build_metric_cost
from .kalman_filter import KalmanFilter
from .dna_track_params import DNATrackParams
from .dna_track import DNATrack


class Tracker:
    def __init__(self, params:DNATrackParams, logger:logging.Logger):
        self.params = params
        self.new_track_overlap_threshold = 0.65

        self.kf = KalmanFilter()
        self.tracks:List[DNATrack] = []
        self._next_id = 1
        self.logger = logger

    def track(self, frame:Frame, detections: List[Detection]) -> Tuple[MatchingSession, List[DNATrack]]:
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
        for tidx, didx in session.matches:
            self.tracks[tidx].update(self.kf, frame, detections[didx], self.params)

        # track의 bounding-box가 exit_region에 포함된 경우는 delete시킨다.
        for track in self.tracks:
            if dna.utils.find_any_centroid_cover(track.location, self.params.exit_zones) >= 0:
                track.mark_deleted()
        
        # unmatch된 track들 중에서 해당 box의 크기가 일정 범위가 넘으면 delete시킴.
        # 그렇지 않은 track은 temporarily lost된 것으로 간주함.
        for tidx in session.unmatched_track_idxes:
            track = self.tracks[tidx]
            if not track.is_deleted():
                if not self.params.is_valid_size(track.location.size()):
                    track.mark_deleted()
                else:
                    track.mark_missed()
                    
        for didx in session.unmatched_strong_det_idxes:
            det = detections[didx]

            # Exit 영역에 포함되는 detection들은 무시한다
            if dna.utils.find_any_centroid_cover(det.bbox, self.params.exit_zones) >= 0:
                continue

            # Stable 영역에 포함되는 detection들은 무시한다.
            # Stable 영역에서는 새로운 track이 생성되지 않도록 함.
            if dna.utils.find_any_centroid_cover(det.bbox, self.params.stable_zones) >= 0:
                continue
            
            # create a new (tentative) track for this unmatched detection
            self._initiate_track(det, frame)

        deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return (session, deleted_tracks)

    def match(self, detections:List[Detection]) -> MatchingSession:
        session = MatchingSession(self.tracks, detections, self.params)
        
        if not(detections and self.tracks):
            # Detection 작업에서 아무런 객체를 검출하지 못한 경우.
            return session

        # 이전 track 객체와 새로 detection된 객체사이의 거리 값을 기준으로 cost matrix를 생성함.
        dist_cost, iou_cost = build_dist_iou_cost(self.kf, self.tracks, detections)

        iou_dist_matcher = IoUDistanceCostMatcher(self.tracks, detections, self.params, iou_cost, dist_cost,
                                                    self.logger.getChild('matcher'))
        matches0 = iou_dist_matcher.match(session.unmatched_track_idxes, session.unmatched_det_idxes)
        if matches0:
            session.update(matches0)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"(motion) hot+tent, strong: {matches_str(self.tracks, matches0)}")
                
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
        ### Appearance를 사용하는 경우는 추적의 안정성을 위해 high-scored detection (즉, strong detection)들과의
        ### matching을 시도한다. 만일 matching시킬 track이 남아 있지만 strong detection이 남아 있지 않는 경우는
        ### 마지막 방법으로 weak detection과 IOU를 통해 match를 시도한다.
        ###########################################################################################################
        if session.unmatched_track_idxes and session.unmatched_strong_det_idxes:
            metric_cost = build_metric_cost(self.tracks, detections, dist_cost, self.params.metric_gate_distance,
                                            session.unmatched_track_idxes, session.unmatched_strong_det_idxes)
            metric_matcher = MetricCostMatcher(self.tracks, detections, self.params, metric_cost, dist_cost, self.logger)
            matches0 = metric_matcher.match(session.unmatched_track_idxes, session.unmatched_strong_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"(metric) all, strong: {matches_str(self.tracks, matches0)}")

        ###########################################################################################################
        ### Match되지 못한 temporarily lost track에 대해서는 motion을 기준으로 재 matching 시킨다.
        ###########################################################################################################
        if session.unmatched_tlost_track_idxes and session.unmatched_det_idxes:
            matches0 = iou_dist_matcher.matcher.match(session.unmatched_tlost_track_idxes, session.unmatched_det_idxes)
            if matches0:
                session.update(matches0)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"tlost, all, {iou_dist_matcher}: {matches_str(self.tracks, matches0)}")

        ###########################################################################################################
        ### 아직 match되지 못한 track이 존재하면, weak detection들과 IoU에 기반한 Hungarian 방식으로 matching을 시도함.
        ###########################################################################################################
        if session.unmatched_track_idxes and session.unmatched_det_idxes:
            iou_last_matcher = HungarianMatcher(iou_cost, self.params.iou_dist_threshold_loose.iou, INVALID_IOU_DISTANCE)
            dist_last_matcher = HungarianMatcher(dist_cost, self.params.iou_dist_threshold_loose.distance, INVALID_DIST_DISTANCE)
            last_resort_matcher = chain(iou_last_matcher, dist_last_matcher)

            if session.unmatched_strong_det_idxes:
                matches0 = last_resort_matcher.match(session.unmatched_track_idxes, session.unmatched_strong_det_idxes)
                if matches0:
                    session.update(matches0)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"all, strong, last_resort[{last_resort_matcher}]: "
                                    f"{matches_str(self.tracks, matches0)}")
                        
        self.revise_matches(iou_dist_matcher.rematcher, session, detections)

        return session

    def _initiate_track(self, detection: Detection, frame:Frame):
        mean, covariance = self.kf.initiate(detection.bbox.to_xyah())
        track = DNATrack(mean, covariance, self._next_id, frame.index, frame.ts,
                            self.params.n_init, self.params.max_age, detection)
        self.tracks.append(track)
        self._next_id += 1

        return track
    
    def revise_matches(self, matcher:Matcher, session:MatchingSession, detection: Detection):
        # tentative track과 strong detection 사이의 match들을 검색한다.
        tent_strong_matches = [m for m in session.matches \
                                    if self.tracks[m[0]].is_tentative() and self.params.is_strong_detection(detection[m[1]])]
        strong_det_idxes = session.unmatched_strong_det_idxes + utils.project(tent_strong_matches, 1)
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
            
    def draw_matched_detections(self, title:str, convas:Image, matches:List[Tuple[int,int]], detections:List[Detection]):
        if matches:
            for t_idx, d_idx in matches:
                label = f'{self.tracks[t_idx].id}({d_idx})'
                det = detections[d_idx]
                if det.score >= self.params.detection_threshold:
                    convas = plot_utils.draw_label(convas, label, Point.from_np(det.bbox.br.astype(int)), color.WHITE, color.BLUE, 1)
                    convas = det.bbox.draw(convas, color.BLUE, line_thickness=1)
                if det.score < self.params.detection_threshold:
                    convas = plot_utils.draw_label(convas, label, Point.from_np(det.bbox.br.astype(int)), color.WHITE, color.RED, 1)
                    convas = det.bbox.draw(convas, color.RED, line_thickness=1)
        cv2.imshow(title, convas)
        cv2.waitKey(1)