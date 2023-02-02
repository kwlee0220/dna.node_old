# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Tuple

import numpy as np
from numpy.linalg import det
from shapely import geometry

import dna
from dna import Box, Size2d, Frame
from dna.detect import Detection
from . import matcher, utils
import kalman_filter
import linear_assignment
import iou_matching
from track import Track
from .matcher import Matcher, ReciprocalCostMatcher

import logging
LOGGER = logging.getLogger('dna.tracker.dnasort')


DIST_THRESHOLD_TIGHT = 20
DIST_THRESHOLD = 50
IOU_THRESHOLD_TIGHT = 0.3
IOU_THRESHOLD = 0.55
IOU_THRESHOLD_LOOSE = 0.8
COMBINED_THRESHOLD_TIGHT = 0.2
# COMBINED_THRESHOLD = 0.5
COMBINED_THRESHOLD_LOOSE = 0.75


class Tracker:
    def __init__(self, domain, detection_threshold:float, metric, params):
        self.domain = domain
        self.metric = metric
        self.detection_threshold = detection_threshold
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

        # Run matching cascade.
        matches, unmatched_track_idxs, unmatched_detections = self._match(detections)

        #########################################################################################################################
        ### kwlee
        import cv2
        convas = frame.image.copy()
        if len(matches) > 0:
            from dna import plot_utils, color
            for t_idx, d_idx in matches:
                label = f'{self.tracks[t_idx].track_id}({d_idx})'
                det = detections[d_idx]
                if det.score >= self.detection_threshold:
                    convas = plot_utils.draw_label(convas, label, det.bbox.br.astype(int), color.WHITE, color.RED, 1)
                    convas = det.bbox.draw(convas, color.RED, line_thickness=1)
                if det.score < self.detection_threshold:
                    convas = plot_utils.draw_label(convas, label, det.bbox.br.astype(int), color.WHITE, color.BLUE, 1)
                    convas = det.bbox.draw(convas, color.BLUE, line_thickness=1)
        cv2.imshow('detections', convas)
        cv2.waitKey(1)
        #########################################################################################################################

        # Update locations of matched tracks
        for tidx, didx in matches:
            self.tracks[tidx].update(self.kf, detections[didx])

        # get bounding-boxes of all tracks
        t_boxes = [utils.track_to_box(track) for track in self.tracks]

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
        matcher.delete_overlapped_tentative_tracks(self.tracks, self.params.max_overlap_ratio)
        
        if unmatched_detections:
            new_track_idxs = unmatched_detections.copy()
            d_boxes: List[Box] = [d.bbox for d in detections]
            matched_det_idxes = utils.project(matches, 1)
            det_idxes = [idx for idx, det in enumerate(detections) 
                            if det.score >= self.detection_threshold or idx in matched_det_idxes]
            
            # 새로 추가될 track의 조건을 만족하지 않는 unmatched_detection들을 제거시킨다.
            for didx in unmatched_detections:
                box = d_boxes[didx]
                
                # 일정크기 이하의 unmatched_detections들은 제외시킴.
                if box.width < self.params.min_size.width or box.height < self.params.min_size.height:
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
                    # LOGGER.debug((f"remove an unmatched detection contained in a blind region: "
                    #                 f"removed={didx}, frame={dna.DEBUG_FRAME_IDX}"))
                    continue

                # 이미 match된 detection과 겹치는 비율이 너무 크거나,
                # 다른 unmatched detection과 겹치는 비율이 크면서 그 detection 영역보다 작은 경우 제외시킴.
                for other_idx, ov in utils.overlap_boxes(box, d_boxes, det_idxes):
                    if other_idx != didx and max(ov) >= self.new_track_overlap_threshold:
                        if other_idx in matched_det_idxes or d_boxes[other_idx].area() >= box.area():
                            LOGGER.debug((f"remove an unmatched detection that overlaps with better one: "
                                            f"removed={didx}, better={other_idx}, ratios={max(ov):.2f}"))
                            new_track_idxs.remove(didx)
                            break

            # 조건을 통과한 unmatched detection들은 새로 생성된 track으로 설정한다.
            for didx in new_track_idxs:
                track = self._initiate_track(detections[didx])
                self.tracks.append(track)
                self._next_id += 1

        deleted_tracks = [t for t in self.tracks if t.is_deleted() and t.age > 1]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in confirmed_tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]

            # # 왜 이전 feature들을 유지하지 않지?
            track.features = [track.features[-1]] #Retain most recent feature of the track.
            # track.features = track.features[-5:]

        active_targets = [t.track_id for t in confirmed_tracks]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

        return deleted_tracks
                
    # 반환값
    #   * Track 작업으로 binding된 track 객체와 할당된 detection 객체: (track_idx, det_idx)
    #   * 기존 track 객체들 중에서 이번 작업에서 확인되지 못한 track 객체들의 인덱스 list
    #   * Track에 할당되지 못한 detection 객체들의 인덱스 list
    def _match(self, detections:List[Detection]) -> Tuple[List[Tuple[int,int]],List[int],List[int]]:
        if len(detections) == 0:
            # Detection 작업에서 아무런 객체를 검출하지 못한 경우.
            return [], utils.all_indices(self.tracks), utils.all_indices(detections)

        # 이전 track 객체와 새로 detection된 객체사이의 거리 값을 기준으로 cost matrix를 생성함.
        dist_cost = self.distance_cost(self.tracks, detections)
        iou_cost = matcher.iou_cost_matrix(self.tracks, detections)
        if dna.DEBUG_PRINT_COST:
            self.print_dist_cost(dist_cost, 999)
            self.print_dist_cost(iou_cost, 1)
        
        matches, unmatched_track_idxes, unmatched_det_idxes = self.match_with_iou_dist(detections, dist_cost, iou_cost)
        if LOGGER.isEnabledFor(logging.INFO) and matches:
            track_ids = [self.tracks[tidx].track_id for tidx in unmatched_track_idxes]
            LOGGER.info(f"(motion-only) hot+tentative, dist+iou: {self.matches_str(matches)}")

        #####################################################################################################
        ### 이 단계까지 오면 지난 frame까지 active하게 추적되던 track들 (hot_track_idxes, tentative_track_idxes)에
        ### 대한 motion 정보만을 통해 matching이 완료됨.
        ### 남은 track들의 경우에는 이전 몇 frame동안 추적되지 못한 track들이어서 motion 정보만으로 matching하기
        ### 어려운 것들만 존재함. 이 track들에 대한 matching을 위해서는 appearance를 사용한 matching을 시도한다.
        ### Appearance를 사용하는 경우는 추적의 안정성을 위해 high-scored detection (즉, strong detection)들과의
        ### matching을 시도한다. 만일 matching시킬 track이 남아 있지만 strong detection이 남아 있지 않는 경우는
        ### 마지막 방법으로 weak detection과 IOU를 통해 match를 시도한다.
        #####################################################################################################
        strong_det_idxes = [idx for idx in unmatched_det_idxes if detections[idx].score >= self.detection_threshold]
        if unmatched_track_idxes and strong_det_idxes:
            matches0, _, strong_det_idxes \
                = self.match_with_metric(detections, dist_cost, unmatched_track_idxes, strong_det_idxes)
            if matches0:
                matches += matches0
                unmatched_track_idxes = utils.subtract(utils.all_indices(self.tracks), utils.project(matches, 0))
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"(motion+metric) all, strong: {self.matches_str(matches0)}")

        #####################################################################################################
        ### 아직 match되지 못한 track이 존재하면, weak detection들과 IoU에 기반한 Hungarian 방식으로 matching을 시도함.
        #####################################################################################################
        if unmatched_track_idxes and strong_det_idxes:
            last_resort_matcher = matcher.HungarianMatcher(iou_cost, IOU_THRESHOLD_LOOSE)
            matches0, _, _, = last_resort_matcher.match(unmatched_track_idxes, strong_det_idxes)
            if matches0:
                matches += matches0
                unmatched_track_idxes = utils.subtract(utils.all_indices(self.tracks), utils.project(matches, 0))
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"all, strong, last_resort[iou={IOU_THRESHOLD_LOOSE}]: {self.matches_str(matches0)}")

        if LOGGER.isEnabledFor(logging.INFO) and matches:
            track_ids = [self.tracks[tidx].track_id for tidx in unmatched_track_idxes]
            LOGGER.info(f"matches={self.matches_str(matches)}, unmodified_tracks={track_ids}, "
                        f"unmodified_detections={strong_det_idxes}")

        return matches, unmatched_track_idxes, strong_det_idxes

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.bbox.to_xyah())
        return Track(mean, covariance, self._next_id, self.params.n_init, self.params.max_age, detection)

    def match_with_metric(self, detections:List[Detection], dist_cost:np.array, unmatched_track_idxes:List[int],
                            strong_det_idxes:List[int]):
        hot_track_idxes = [idx for idx in unmatched_track_idxes \
                                if self.tracks[idx].is_confirmed() and self.tracks[idx].time_since_update <= 3]
        tentative_track_idxes = [idx for idx in unmatched_track_idxes if not self.tracks[idx].is_confirmed()]
        matches = []

        #####################################################################################################
        ########## 통합 비용 행렬을 생성한다.
        ########## 통합 비용을 계산할 때는 weak detection은 제외시킨다.
        ########## cmatrix: 통합 비용 행렬
        ########## ua_matrix: unconfirmed track을 고려한 통합 비용 행렬
        #####################################################################################################
        # metric_cost = self.metric_cost(self.tracks, detections)
        metric_cost = self.metric_cost(self.tracks, detections, unmatched_track_idxes, strong_det_idxes)
        dist_metric_cost, cmask = matcher.combine_cost_matrices(metric_cost, dist_cost, self.tracks, detections)
        dist_metric_cost[cmask] = 9.99
        if dna.DEBUG_PRINT_COST:
            matcher.print_matrix(self.tracks, detections, metric_cost, 1, unmatched_track_idxes, strong_det_idxes)
            matcher.print_matrix(self.tracks, detections, dist_metric_cost, 9.98, unmatched_track_idxes, strong_det_idxes)

        hung_matcher = matcher.HungarianMatcher(dist_metric_cost, None)

        #####################################################################################################
        ################ Hot track에 한정해서 강한 threshold를 사용해서  matching 실시
        ################ Tentative track에 비해 2배 이상 먼거리를 갖는 경우에는 matching을 하지 않도록 함.
        #####################################################################################################
        if hot_track_idxes and strong_det_idxes:
            if dna.DEBUG_PRINT_COST:
                matcher.print_matrix(self.tracks, detections, dist_metric_cost, COMBINED_THRESHOLD_TIGHT,
                                        hot_track_idxes, strong_det_idxes)
            matches0, _, strong_det_idxes \
                = hung_matcher.match_threshold(COMBINED_THRESHOLD_TIGHT, hot_track_idxes, strong_det_idxes)
            if matches0:
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"hot, strong, combined_tight: {self.matches_str(matches0)}")
                matches += matches0
                unmatched_track_idxes = utils.subtract(unmatched_track_idxes, utils.project(matches0, 0))

        #####################################################################################################
        ################ Tentative track에 한정해서 강한 threshold를 사용해서  matching 실시
        #####################################################################################################
        if tentative_track_idxes and strong_det_idxes:
            matches0, _, strong_det_idxes \
                = hung_matcher.match_threshold(COMBINED_THRESHOLD_TIGHT, tentative_track_idxes, strong_det_idxes)
            if matches0:
                if dna.DEBUG_PRINT_COST:
                    LOGGER.debug(f"tentative, strong, combined_tight: {self.matches_str(matches0)}")
                matches += matches0
                unmatched_track_idxes = utils.subtract(unmatched_track_idxes, utils.project(matches0, 0))

        #####################################################################################################
        ################ 전체 track에 대해 matching 실시
        #####################################################################################################
        if unmatched_track_idxes and strong_det_idxes:
            if dna.DEBUG_PRINT_COST:
                matcher.print_matrix(self.tracks, detections, dist_metric_cost, COMBINED_THRESHOLD_LOOSE,
                                        unmatched_track_idxes, strong_det_idxes)

            matches0, unmatched_track_idxes, strong_det_idxes \
                = hung_matcher.match_threshold(COMBINED_THRESHOLD_LOOSE, unmatched_track_idxes, strong_det_idxes)
            if matches0:
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"all, strong, combined_normal): {self.matches_str(matches0)}")
                matches += matches0

        return matches, unmatched_track_idxes, strong_det_idxes

    ###############################################################################################################
    # kwlee
    # def metric_cost(self, tracks, detections):
    #     features = np.array([det.feature for det in detections])
    #     targets = np.array([track.track_id for track in tracks])
    #     return self.metric.distance(features, targets)

    def metric_cost(self, tracks:List[Track], detections:List[Detection], track_idxes, det_idxes):
        targets = [tracks[idx].track_id for idx in track_idxes]
        features = np.array([detections[idx].feature for idx in det_idxes])
        reduced_matrix = self.metric.distance(features, targets)

        cost_matrix = np.ones((len(tracks), len(detections)))
        for row_idx, t_idx in enumerate(track_idxes):
            for col_idx, d_idx in enumerate(det_idxes):
                cost_matrix[t_idx, d_idx] = reduced_matrix[row_idx, col_idx]
        return cost_matrix

    # kwlee
    def distance_cost(self, tracks, detections):
        dist_matrix = np.zeros((len(tracks), len(detections)))
        if len(tracks) > 0 and len(detections) > 0:
            measurements = np.asarray([det.bbox.to_xyah() for det in detections])
            for row, track in enumerate(tracks):
                mahalanovis_dist = self.kf.gating_distance(track.mean, track.covariance, measurements)
                dist_matrix[row, :] = mahalanovis_dist * (1 + 0.75*(track.time_since_update-1))
                # dist_matrix[row, :] = self.kf.gating_distance(track.mean, track.covariance, measurements)
        return dist_matrix

    def matches_str(self, matches):
        return ",".join([f"({self.tracks[tidx].track_id}, {didx})" for tidx, didx in matches])

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

    
    def match_with_iou_dist(self, detections:List[Detection], dist_cost:np.array, iou_cost:np.array):
        hot_track_idxes = [i for i, t in enumerate(self.tracks) if t.is_confirmed() and t.time_since_update <= 3]
        tentative_track_idxes = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        strong_det_idxes = [idx for idx, det in enumerate(detections) if det.score >= self.detection_threshold]
        weak_det_idxes = [idx for idx, det in enumerate(detections) if det.score < self.detection_threshold]
        matches = []

        #####################################################################################################
        ### Hot track과 tentative track들에 한정해서 tight한 distance threshold를 사용해서 matching 실시.
        ### Hot track과 tentative track들은 바로 이전 frame에서 성공적으로 detection과 binding되었기 때문에,
        ### 이번 frame에서의 position prediction이 상당히 정확할 것으로 예상되므로, 이들과 아주 가깝게
        ### 위치한 detection들은 해당 track이 확률이 매우 크다.
        #####################################################################################################
        tight_iou_dist_matcher = Matcher.chain(ReciprocalCostMatcher(iou_cost, IOU_THRESHOLD_TIGHT),
                                                ReciprocalCostMatcher(dist_cost, DIST_THRESHOLD_TIGHT))
        matches0, _, strong_det_idxes = tight_iou_dist_matcher.match(hot_track_idxes+tentative_track_idxes, strong_det_idxes)
        if matches0:
            matches += matches0
            matched_track_idxes = utils.project(matches, 0)
            hot_track_idxes = utils.subtract(hot_track_idxes, matched_track_idxes)
            tentative_track_idxes = utils.subtract(tentative_track_idxes, matched_track_idxes)
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"(very-tight) hot+teta, strong, "
                            f"iou[{IOU_THRESHOLD_TIGHT}]+dist[{DIST_THRESHOLD_TIGHT}]): {self.matches_str(matches0)}")
            
        #####################################################################################################
        ### Hot track에 한정해서 distance 정보를 사용해서 matching 실시.
        #####################################################################################################
        iou_matcher = ReciprocalCostMatcher(iou_cost, IOU_THRESHOLD)
        dist_matcher = ReciprocalCostMatcher(dist_cost, DIST_THRESHOLD)
        iou_dist_matcher = Matcher.chain(iou_matcher, dist_matcher)

        #----------------------------------------------------------------------------------------------------
        # 각 hot track을 기준으로 다음과 같은 조건을 만족하는 high-scored (=strong) detection과 matching시킨다.
        #----------------------------------------------------------------------------------------------------
        matches0, hot_track_idxes, strong_det_idxes = iou_dist_matcher.match(hot_track_idxes, strong_det_idxes)
        if matches0:
            matches += matches0
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"hot, strong, iou({IOU_THRESHOLD})+dist({DIST_THRESHOLD}): {self.matches_str(matches0)}")
        
        #----------------------------------------------------------------------------------------------------
        # Hot track 중에서 match되지 못한 것이 존재하는 경우에는 weak detection들과 match를 시도한다.
        #----------------------------------------------------------------------------------------------------
        if hot_track_idxes and weak_det_idxes:
            matches0, hot_track_idxes, weak_det_idxes = iou_dist_matcher.match(hot_track_idxes, weak_det_idxes)
            if matches0:
                matches += matches0
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"hot, weak, iou({IOU_THRESHOLD})+dist({DIST_THRESHOLD}): {self.matches_str(matches0)}")
            
        #####################################################################################################
        ### Tentative track에 한정해서 tight한 distance 정보를 사용해서 matching 실시.
        ### Tentative track도 바로 이전 frame까지도 track되던 이동체이기 때문에 motion 정보만 활용한 matching을 실시함
        #####################################################################################################
        if tentative_track_idxes and strong_det_idxes:
            matches0, tentative_track_idxes, strong_det_idxes \
                = iou_dist_matcher.match(tentative_track_idxes, strong_det_idxes)
            if matches0:
                matches += matches0
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"tentative, strong, iou({IOU_THRESHOLD})+dist({DIST_THRESHOLD}): "
                                f"{self.matches_str(matches0)}")
        #----------------------------------------------------------------------------------------------------
        # Tentative track 중에서 match되지 못한 것이 존재하는 경우에는 weak detection들과 match를 시도한다.
        # 다만, ghost detection에 따른 track 생성의 가능성을 낮추기 위해 weak detection들과는 tight한 distance를 사용함.
        #----------------------------------------------------------------------------------------------------
        if tentative_track_idxes and weak_det_idxes:
            matches0, tentative_track_idxes, weak_det_idxes \
                = tight_iou_dist_matcher.match(tentative_track_idxes, weak_det_idxes)
            if matches0:
                matches += matches0
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f"tentative, weak, iou({IOU_THRESHOLD_TIGHT})+dist({DIST_THRESHOLD_TIGHT}): "
                                f"{self.matches_str(matches0)}")
            
        #####################################################################################################
        ###  지금까지 match되지 못한 strong detection들 중에서 이미 matching된 detection들과 많이 겹치는 경우,
        ###  해당 detection을 제외시킨다. 이를 통해 실체 이동체 주변에 여러개 잡히는 ghost detection으로 인한
        ###  새 track 생성 가능성을 낮춘다.
        #####################################################################################################
        if strong_det_idxes:
            d_boxes = [det.bbox for det in detections]
            matched_boxes = [d_boxes[idx] for idx in utils.project(matches, 1)]
            for m_box in matched_boxes:
                strong_det_idxes = [bidx for bidx in strong_det_idxes
                                            if max(m_box.overlap_ratios(d_boxes[bidx])) <= self.params.max_overlap_ratio]
                if len(strong_det_idxes) == 0:
                    break

        unmatched_track_idxes = utils.subtract(utils.all_indices(self.tracks), utils.project(matches, 0))
        # if LOGGER.isEnabledFor(logging.INFO) and matches:
        #     track_ids = [self.tracks[tidx].track_id for tidx in unmatched_track_idxes]
        #     LOGGER.info(f"(motion-only) hot+tentative, dist+iou: {self.matches_str(matches)}")

        return matches, unmatched_track_idxes, strong_det_idxes+weak_det_idxes