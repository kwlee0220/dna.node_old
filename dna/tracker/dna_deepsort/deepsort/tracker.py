# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from numpy.linalg import det
from shapely import geometry

import dna
from dna import Box, Size2d, Frame
from dna.detect import Detection
from dna.tracker import DNASORTParams, utils, IouDistThreshold
from dna.tracker.matcher import Matcher, MatchingSession, chain, matches_str, match_str, \
                                IoUDistanceCostMatcher, MetricCostMatcher, HungarianMatcher, \
                                INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE, INVALID_METRIC_DISTANCE
from .matcher import print_matrix
import kalman_filter
from track import Track

import logging
LOGGER = logging.getLogger('dna.tracker.dnasort')

def remove_weak_overlaps(detections:List[Detection], params:DNASORTParams):
    boxes = [det.bbox for det in detections]
    strong_boxes = [det.bbox for det in detections if params.is_strong_detection(det)]
    weak_idxes = {i for i, det in enumerate(detections) if not params.is_strong_detection(det)}

    if weak_idxes:
        for s_box in strong_boxes:
            overlapped_idxes = {w_idx for w_idx in weak_idxes if s_box.iou(boxes[w_idx]) > params.overlap_supress_ratio}
            if overlapped_idxes:
                weak_idxes = weak_idxes.difference(overlapped_idxes)

        return [det for i, det in enumerate(detections) if i in weak_idxes or params.is_strong_detection(det)]
    else:
        return detections


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

        # Run matching cascade.
        matches, unmatched_track_idxs, unmatched_detections = self._match(detections)
        if LOGGER.isEnabledFor(logging.DEBUG):
            bindings = [(self.tracks[t_idx].track_id, d_idx) for t_idx, d_idx in matches]
            LOGGER.debug(f'matches={bindings}, unmatched: tracks={unmatched_track_idxs}, detections={unmatched_detections}')

        if dna.DEBUG_SHOW_IMAGE:
            import cv2
            convas = frame.image.copy()
            if len(matches) > 0:
                from dna import plot_utils, color
                for t_idx, d_idx in matches:
                    label = f'{self.tracks[t_idx].track_id}({d_idx})'
                    det = detections[d_idx]
                    if det.score >= self.params.detection_threshold:
                        convas = plot_utils.draw_label(convas, label, det.bbox.br.astype(int), color.WHITE, color.RED, 1)
                        convas = det.bbox.draw(convas, color.RED, line_thickness=1)
                    if det.score < self.params.detection_threshold:
                        convas = plot_utils.draw_label(convas, label, det.bbox.br.astype(int), color.WHITE, color.BLUE, 1)
                        convas = det.bbox.draw(convas, color.BLUE, line_thickness=1)
            cv2.imshow('detections', convas)
            cv2.waitKey(1)

        # Update locations of matched tracks
        for tidx, didx in matches:
            self.tracks[tidx].update(self.kf, detections[didx], self.params)

        # get bounding-boxes of all tracks
        t_boxes = [track.to_box() for track in self.tracks]

        # track의 bounding-box가 exit_region에 포함된 경우는 delete시킨다.
        for tidx in range(len(self.tracks)):
            if dna.utils.find_any_centroid_cover(t_boxes[tidx], self.params.exit_zones) >= 0:
                self.tracks[tidx].mark_deleted()
        
        # unmatch된 track들 중에서 해당 box가 이미지 영역에서 1/4이상 넘어가면
        # lost된 것으로 간주한다.
        for tidx in unmatched_track_idxs:
            track:Track = self.tracks[tidx]
            if not track.is_deleted():
                ratios = t_boxes[tidx].overlap_ratios(self.domain)
                if ratios[0] < 0.85:
                    track.mark_deleted()
                else:
                    track.mark_missed()

        # confirmed track과 너무 가까운 tentative track들을 제거한다.
        # 일반적으로 이런 track들은 이전 frame에서 한 물체의 여러 detection 검출을 통해 track이 생성된
        # 경우가 많아서 이를 제거하기 위함이다.
        delete_overlapped_tentative_tracks(self.tracks, self.params.overlap_supress_ratio)
        
        if unmatched_detections:
            new_track_idxs = unmatched_detections.copy()
            d_boxes: List[Box] = [d.bbox for d in detections]
            matched_det_idxes = utils.project(matches, 1)
            det_idxes = [idx for idx, det in enumerate(detections) 
                            if det.score >= self.params.detection_threshold or idx in matched_det_idxes]
            
            # 새로 추가될 track의 조건을 만족하지 않는 unmatched_detection들을 제거시킨다.
            for didx in unmatched_detections:
                box = d_boxes[didx]
                
                # 일정크기 이하의 unmatched_detections들은 제외시킴.
                if box.width < self.params.min_new_track_size.width \
                    or box.height < self.params.min_new_track_size.height:
                    new_track_idxs.remove(didx)
                    continue

                # Stable 영역에 포함되는 detection들은 무시한다.
                # Stable 영역에서는 새로운 track이 생성되지 않도록 함.
                if dna.utils.find_any_centroid_cover(box, self.params.stable_zones) >= 0:
                    new_track_idxs.remove(didx)
                    continue

                # Exit 영역에 포함되는 detection들은 무시한다
                if dna.utils.find_any_centroid_cover(box, self.params.exit_zones) >= 0:
                    new_track_idxs.remove(didx)
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        LOGGER.debug((f"remove an unmatched detection contained in a blind region: "
                                        f"removed={didx}, frame={frame.index}"))
                    continue

                # 이미 match된 detection과 겹치는 비율이 너무 크거나,
                # 다른 unmatched detection과 겹치는 비율이 크면서 그 detection 영역보다 작은 경우 제외시킴.
                for other_idx, ov in overlap_boxes(box, d_boxes, det_idxes):
                    if other_idx != didx and max(ov) >= self.new_track_overlap_threshold:
                        if other_idx in matched_det_idxes or d_boxes[other_idx].area() >= box.area():
                            LOGGER.debug((f"remove overlapped detection: removed({didx}), better({other_idx}), "
                                            f"ratios={max(ov):.2f}"))
                            new_track_idxs.remove(didx)
                            break

            # 조건을 통과한 unmatched detection들은 새로 생성된 track으로 설정한다.
            for didx in new_track_idxs:
                track = self._initiate_track(detections[didx])
                self.tracks.append(track)
                self._next_id += 1

        deleted_tracks = [t for t in self.tracks if t.is_deleted() and t.age > 1]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return deleted_tracks

                
    def _match(self, detections:List[Detection]) -> Tuple[List[Tuple[int,int]],List[int],List[int]]:
        if not(detections and self.tracks):
            # Detection 작업에서 아무런 객체를 검출하지 못한 경우.
            return [], utils.all_indices(self.tracks), utils.all_indices(detections)

        ###########################################################################################################
        ### 이전 track 객체와 새로 detection된 객체사이의 거리 값을 기준으로 cost matrix를 생성함.
        ###########################################################################################################
        iou_cost, dist_cost = self.build_iou_dist_cost(detections)
        if dna.DEBUG_PRINT_COST:
            self.print_dist_cost(iou_cost, 1)
            self.print_dist_cost(dist_cost, 999)

        session = MatchingSession(self.tracks, detections, self.params)

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
                        
        self._yield_strong_tentative_matches(iou_dist_matcher.matcher, session, detections)

        if session.unmatched_strong_det_idxes:
            self._rematch(iou_dist_matcher.matcher, session, detections)

        return session.matches, session.unmatched_track_idxes, session.unmatched_strong_det_idxes
    
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

def delete_overlapped_tentative_tracks(tracks, threshold):
    confirmed_track_idxs = [i for i, t in enumerate(tracks) 
                                if t.is_confirmed() and t.time_since_update == 1 and not t.is_deleted()]
    unconfirmed_track_idxs = [i for i, t in enumerate(tracks)
                                if not t.is_confirmed() and not t.is_deleted()]
    if not (confirmed_track_idxs and unconfirmed_track_idxs):
        return

def overlap_boxes(target:Box, boxes:List[Box], box_indices:List[int]=None) \
    -> List[Tuple[int, Tuple[float,float,float]]]:
    if box_indices is None:
        return ((idx, target.overlap_ratios(box)) for idx, box in enumerate(boxes))
    else:
        return ((idx, target.overlap_ratios(boxes[idx])) for idx in box_indices)