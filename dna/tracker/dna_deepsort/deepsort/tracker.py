# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Tuple, Set

import numpy as np
import numpy.typing as npt
from numpy.linalg import det
import cv2

import dna
from dna import Box, Size2d, Frame, plot_utils, color, Image
from dna.detect import Detection
from dna.tracker import DNASORTParams, utils, IouDistThreshold
from dna.tracker.matcher import Matcher, MatchingSession, chain, matches_str, match_str, \
                                IoUDistanceCostMatcher, MetricCostMatcher, HungarianMatcher, \
                                INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE, INVALID_METRIC_DISTANCE
import kalman_filter
from track import Track

import logging
LOGGER = logging.getLogger('dna.tracker.dnasort')


class Tracker:
    def __init__(self, domain, metric, params:DNASORTParams):
        self.domain = domain
        self.metric = metric
        self.params = params
        self.new_track_overlap_threshold = 0.65

        self.kf = kalman_filter.KalmanFilter()
        self.tracks:List[Track] = []
        self._next_id = 1

        self.current_frame = None

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections:List[Detection], frame: Frame):
        self.current_frame = frame
        dna.DEBUG_FRAME_INDEX = frame.index
        
        # Detection box 크기에 따라 invalid한 detection들을 제거한다.
        detections = [det for det in detections if self.params.is_valid_detection(det)]
        
        if dna.DEBUG_SHOW_IMAGE:
            self.draw_detections('detections', frame.image.copy(), detections)

        # Run matching cascade.
        matches, unmatched_track_idxs, unmatched_detections = self.match(detections)
        if LOGGER.isEnabledFor(logging.DEBUG):
            bindings = [(self.tracks[t_idx].track_id, d_idx) for t_idx, d_idx in matches]
            LOGGER.debug(f'matches={bindings}, unmatched: tracks={unmatched_track_idxs}, detections={unmatched_detections}')

        if dna.DEBUG_SHOW_IMAGE:
            self.draw_matched_detections("detections", frame.image.copy(), matches, detections)

        # Update locations of matched tracks
        for tidx, didx in matches:
            self.tracks[tidx].update(self.kf, detections[didx], self.params)

        # track의 bounding-box가 exit_region에 포함된 경우는 delete시킨다.
        for track in self.tracks:
            if dna.utils.find_any_centroid_cover(track.to_box(), self.params.exit_zones) >= 0:
                track.mark_deleted()
        
        # unmatch된 track들 중에서 해당 box가 이미지 영역에서 1/4이상 넘어가면
        # lost된 것으로 간주한다.
        for tidx in unmatched_track_idxs:
            track = self.tracks[tidx]
            if not track.is_deleted():
                ratios = track.to_box().overlap_ratios(self.domain)
                if ratios[0] < 0.85:
                    track.mark_deleted()
                else:
                    track.mark_missed()
                    
        for didx in unmatched_detections:
            box = detections[didx].bbox

            # Exit 영역에 포함되는 detection들은 무시한다
            if dna.utils.find_any_centroid_cover(box, self.params.exit_zones) >= 0:
                continue

            # Stable 영역에 포함되는 detection들은 무시한다.
            # Stable 영역에서는 새로운 track이 생성되지 않도록 함.
            if dna.utils.find_any_centroid_cover(box, self.params.stable_zones) >= 0:
                continue
            
            track = self._initiate_track(detections[didx])
            self.tracks.append(track)
            self._next_id += 1

        deleted_tracks = [t for t in self.tracks if t.is_deleted() and t.age > 1]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return deleted_tracks

    def match(self, detections:List[Detection]) -> Tuple[List[Tuple[int,int]],List[int],List[int]]:
        session = MatchingSession(self.tracks, detections, self.params)
        
        if not(detections and self.tracks):
            # Detection 작업에서 아무런 객체를 검출하지 못한 경우.
            return session.matches, session.unmatched_track_idxes, session.unmatched_strong_det_idxes

        ###########################################################################################################
        ### 이전 track 객체와 새로 detection된 객체사이의 거리 값을 기준으로 cost matrix를 생성함.
        ###########################################################################################################
        iou_cost, dist_cost = self.build_iou_dist_cost(detections)
        if dna.DEBUG_PRINT_COST:
            self.print_dist_cost(iou_cost, 1)
            self.print_dist_cost(dist_cost, 999)

        iou_dist_matcher = IoUDistanceCostMatcher(self.tracks, detections, self.params, iou_cost, dist_cost, LOGGER)
        matches0 = iou_dist_matcher.match(session.unmatched_track_idxes, session.unmatched_det_idxes)
        if matches0:
            session.update(matches0)
                
        ###########################################################################################################
        ###  지금까지 match되지 못한 strong detection들 중에서 이미 matching된 detection들과 많이 겹치는 경우,
        ###  해당 detection을 제외시킨다. 이를 통해 실체 이동체 주변에 여러개 잡히는 ghost detection으로 인한
        ###  새 track 생성 가능성을 낮춘다.
        ###########################################################################################################
        if session.unmatched_strong_det_idxes:
            d_boxes = [det.bbox for det in detections]
            def select_overlaps(box:Box, candidates:List[int]):
                return [idx for idx in candidates if box.iou(d_boxes[idx]) >= self.params.overlap_supress_ratio]
            
            for d_idx, m_box in ((i, d_boxes[i]) for i in utils.project(session.matches, 1)):
                overlap_det_idxes = select_overlaps(m_box, session.unmatched_strong_det_idxes)
                session.remove_det_idxes(overlap_det_idxes)
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
            metric_cost = self.build_metric_cost(detections, dist_cost, session.unmatched_track_idxes,
                                                 session.unmatched_strong_det_idxes)
            metric_matcher = MetricCostMatcher(self.tracks, detections, self.params, metric_cost, dist_cost, LOGGER)
            matches0 = metric_matcher.match(session.unmatched_track_idxes, session.unmatched_strong_det_idxes)
            if matches0:
                session.update(matches0)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"(metric) all, strong: {matches_str(self.tracks, matches0)}")

        if session.unmatched_tlost_track_idxes and session.unmatched_det_idxes:
            matches0 = iou_dist_matcher.matcher.match(session.unmatched_tlost_track_idxes, session.unmatched_det_idxes)
            if matches0:
                session.update(matches0)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"tlost, all, {iou_dist_matcher}: {matches_str(self.tracks, matches0)}")

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
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        LOGGER.debug(f"all, strong, last_resort[{last_resort_matcher}]: "
                                    f"{matches_str(self.tracks, matches0)}")
                        
        # if session.unmatched_confirmed_track_idxes or session.unmatched_strong_det_idxes:
        self.revise_matches(iou_dist_matcher.matcher, session, detections)
        # self._yield_strong_tentative_matches(iou_dist_matcher.matcher, session, detections)

        # if session.unmatched_strong_det_idxes:
        #     self._rematch(iou_dist_matcher.matcher, session, detections)

        return session.matches, session.unmatched_track_idxes, session.unmatched_strong_det_idxes
    
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
                if LOGGER.isEnabledFor(logging.DEBUG):
                    old_weak_str = [match_str(self.tracks, old_weak_match)] if old_weak_match else []
                    old_strong_str = [match_str(self.tracks, old_strong_match)] if old_strong_match else []
                    old_str = ', '.join(old_weak_str+old_strong_str)
                    LOGGER.debug(f'rematch: {old_str} -> {match_str(self.tracks, match)}')
                          
            matches1 = matcher.match(deprived_track_idxes, session.unmatched_det_idxes)
            if matches1:
                session.update(matches1)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"rematch yielding tracks, {matcher}: {matches_str(self.tracks, matches1)}")
    
    def _yield_strong_tentative_matches(self, matcher:Matcher, session:MatchingSession,  detection: Detection):
        # strong detection들과 match된 tentative track을 검색함.
        strong_tentative_matches = [m for m in session.matches \
                                        if self.tracks[m[0]].is_tentative() and self.params.is_strong_detection(detection[m[1]])]
        # weak detection들과 match된 non-tentative track들을 검색함
        weak_track_idxes = [t_idx for t_idx, d_idx in session.matches \
                                    if not self.tracks[t_idx].is_tentative() and not self.params.is_strong_detection(detection[d_idx])]
        
        if strong_tentative_matches and weak_track_idxes:
            strong_det_idxes = utils.project(strong_tentative_matches, 1)
            
            matches0 = matcher.match(weak_track_idxes, strong_det_idxes)
            if matches0:
                for match in matches0:
                    old_weak_match = session.find_match_by_track(match[0])
                    if old_weak_match:
                        old_tentative_match = session.find_match_by_det(match[1])
                        session.pull_out(old_weak_match)
                        session.pull_out(old_tentative_match)
                        session.update([match])
                        if LOGGER.isEnabledFor(logging.DEBUG):
                            LOGGER.debug(f'yield detection: {match_str(self.tracks, old_tentative_match)} -> {match_str(self.tracks, match)}')
                            
                matches1 = matcher.match(utils.project(strong_tentative_matches, 0), session.unmatched_strong_det_idxes)
                if matches1:
                    session.update(matches1)
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        LOGGER.debug(f"rematch yielding tracks, {matcher}: {matches_str(self.tracks, matches1)}")

    def _rematch(self, matcher:Matcher, session:MatchingSession,  detection: Detection):
        # weak detection들과 match된 track들만 뽑아서 unmatch된 strong detection들과 다시 match를 시도함.
        weak_track_idxes = [t_idx for t_idx, d_idx in session.matches if not self.params.is_strong_detection(detection[d_idx])]

        matches0 = matcher.match(weak_track_idxes, session.unmatched_strong_det_idxes)
        for match in matches0:
            if old_match := session.find_match_by_track(match[0]):
                session.pull_out(old_match)
                session.update([match])
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f're-match: {match_str(self.tracks, old_match)} -> {match_str(self.tracks, match)}')

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.bbox.to_xyah())
        return Track(mean, covariance, self._next_id, self.params.n_init, self.params.max_age, detection)
            
    def draw_matched_detections(self, title:str, convas:Image, matches:List[Tuple[int,int]], detections:List[Detection]):
        if matches:
            for t_idx, d_idx in matches:
                label = f'{self.tracks[t_idx].track_id}({d_idx})'
                det = detections[d_idx]
                if det.score >= self.params.detection_threshold:
                    convas = plot_utils.draw_label(convas, label, det.bbox.br.astype(int), color.WHITE, color.BLUE, 1)
                    convas = det.bbox.draw(convas, color.RED, line_thickness=1)
                if det.score < self.params.detection_threshold:
                    convas = plot_utils.draw_label(convas, label, det.bbox.br.astype(int), color.WHITE, color.RED, 1)
                    convas = det.bbox.draw(convas, color.BLUE, line_thickness=1)
        cv2.imshow(title, convas)
        cv2.waitKey(1)

    def draw_detections(self, title:str, convas:Image, detections:List[Detection], line_thickness=1):
        for idx, det in enumerate(detections):
            if det.score < self.params.detection_threshold:
                convas = plot_utils.draw_label(convas, str(idx), det.bbox.br.astype(int), color.WHITE, color.RED, 1)
                convas = det.bbox.draw(convas, color.RED, line_thickness=line_thickness) 
        for idx, det in enumerate(detections):
            if det.score >= self.params.detection_threshold:
                convas = plot_utils.draw_label(convas, str(idx), det.bbox.br.astype(int), color.WHITE, color.BLUE, 1)
                convas = det.bbox.draw(convas, color.BLUE, line_thickness=line_thickness)
        cv2.imshow(title, convas)
        cv2.waitKey(1)

    ###############################################################################################################
    # kwlee

    def build_iou_dist_cost(self, detections:List[Detection]) -> Tuple[np.ndarray, np.ndarray]:
        iou_cost = self._build_iou_cost(self.tracks, detections)
        dist_cost = self._build_dist_cost(self.tracks, detections)
        
        bad_ratio_mask = ~self._build_task_det_ratio_mask(detections)
        iou_cost[bad_ratio_mask] = 1
        dist_cost[bad_ratio_mask] = INVALID_DIST_DISTANCE

        return iou_cost, dist_cost
    
    def _build_iou_cost(self, tracks:List[Track], detections:List[Detection]) -> np.ndarray:
        matrix = np.zeros((len(tracks), len(detections)))
        for t_idx, track in enumerate(tracks):
            t_box = track.to_box()
            for d_idx, det in enumerate(detections):
                matrix[t_idx,d_idx] = 1 - t_box.iou(det.bbox)
        return matrix

    def _build_dist_cost(self, tracks:List[Track], detections:List[Detection]) -> np.ndarray:
        dist_matrix = np.zeros((len(tracks), len(detections)))
        if tracks and detections:
            measurements = np.asarray([det.bbox.to_xyah() for det in detections])
            for t_idx, track in enumerate(tracks):
                mahalanovis_dist = self.kf.gating_distance(track.mean, track.covariance, measurements)
                dist_matrix[t_idx, :] = mahalanovis_dist * (1 + 0.75*(track.time_since_update-1))
        return dist_matrix

    _AREA_RATIO_LIMITS = (0.3, 2.8)
    def _build_task_det_ratio_mask(self, detections, area_ratio_limits:npt.ArrayLike=_AREA_RATIO_LIMITS):
        det_areas = np.array([det.bbox.area() for det in detections])
        
        area_ratio_limits = np.array(area_ratio_limits)
        mask = np.zeros((len(self.tracks), len(detections)), dtype=bool)
        for t_idx, track in enumerate(self.tracks):
            limits = area_ratio_limits * track.to_box().area()
            mask[t_idx,:] = (det_areas >= limits[0]) & (det_areas <= limits[1])
            
        return mask

    def build_metric_cost(self, detections:List[Detection], dist_cost:np.ndarray, track_idxes, det_idxes):
        metric_cost = self._build_raw_metric_cost(self.tracks, detections, track_idxes, det_idxes)
        self._gate_metric_cost(metric_cost, dist_cost, self.tracks, detections, self.params.metric_gate_distance)

        return metric_cost

    def _build_raw_metric_cost(self, tracks:List[Track], detections:List[Detection],
                            track_idxes:List[int], det_idxes:List[int]) -> np.ndarray:
        def build_matrix(tracks:List[Track], features) -> np.ndarray:
            cost_matrix = np.zeros((len(tracks), len(features)))
            for i, track in enumerate(tracks):
                samples = track.features
                if samples and len(features) > 0:
                    cost_matrix[i, :] = self.metric._metric(samples, features)
            return cost_matrix

        reduced_tracks = list(utils.get_items(tracks, track_idxes))
        reduced_features = list(det.feature for det in utils.get_items(detections, det_idxes))
        reduced_matrix = build_matrix(reduced_tracks, reduced_features)

        cost_matrix = np.ones((len(tracks), len(detections)))
        for row_idx, t_idx in enumerate(track_idxes):
            for col_idx, d_idx in enumerate(det_idxes):
                cost_matrix[t_idx, d_idx] = reduced_matrix[row_idx, col_idx]
        return cost_matrix

    def _gate_metric_cost(self, metric_costs:np.ndarray, dist_costs:np.ndarray,
                         tracks:List[Track], detections:List[Detection],
                         gate_threshold:float) -> None:
        for t_idx, track in enumerate(tracks):
            t_box = track.to_box()
            for d_idx, det in enumerate(detections):
                if dist_costs[t_idx, d_idx] == INVALID_DIST_DISTANCE:
                    metric_costs[t_idx, d_idx] = INVALID_METRIC_DISTANCE
                elif dist_costs[t_idx, d_idx] > gate_threshold:
                    box_dist = t_box.min_distance_to(det.bbox)
                    if box_dist > 150:
                        metric_costs[t_idx, d_idx] = INVALID_METRIC_DISTANCE

    def print_dist_cost(self, dist_cost, trim_overflow=None):
        if trim_overflow:
            dist_cost = dist_cost.copy()
            dist_cost[dist_cost > trim_overflow] = trim_overflow

        for tidx, track in enumerate(self.tracks):
            dists = [int(round(v)) for v in dist_cost[tidx]]
            track_str = f" {tidx:02d}: {track.track_id:03d}({track.state},{track.time_since_update:02d})"
            dist_str = ', '.join([f"{v:4d}" if v != trim_overflow else "    " for v in dists])
            print(f"{track_str}: {dist_str}")

    ###############################################################################################################

def overlap_boxes(target:Box, boxes:List[Box], box_indices:List[int]=None) \
    -> List[Tuple[int, Tuple[float,float,float]]]:
    if box_indices is None:
        return ((idx, target.overlap_ratios(box)) for idx, box in enumerate(boxes))
    else:
        return ((idx, target.overlap_ratios(boxes[idx])) for idx in box_indices)