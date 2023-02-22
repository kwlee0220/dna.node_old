from __future__ import absolute_import
from typing import List, Tuple, Set

import numpy as np
import numpy.typing as npt

import dna
from dna.detect import Detection
from dna.tracker import ObjectTrack, utils
from ..kalman_filter import KalmanFilter
from .base import INVALID_DIST_DISTANCE, INVALID_IOU_DISTANCE, INVALID_METRIC_DISTANCE


def build_dist_cost(kf:KalmanFilter, tracks:List[ObjectTrack], detections:List[Detection]) -> np.ndarray:
    dist_matrix = np.zeros((len(tracks), len(detections)))
    if tracks and detections:
        measurements = np.asarray([det.bbox.to_xyah() for det in detections])
        for t_idx, track in enumerate(tracks):
            mahalanovis_dist = kf.gating_distance(track.mean, track.covariance, measurements)
            dist_matrix[t_idx, :] = mahalanovis_dist * (1 + 0.75*(track.time_since_update-1))
    return dist_matrix


def build_iou_cost(tracks:List[ObjectTrack], detections:List[Detection]) -> np.ndarray:
    matrix = np.zeros((len(tracks), len(detections)))
    for t_idx, track in enumerate(tracks):
        t_box = track.location
        for d_idx, det in enumerate(detections):
            matrix[t_idx,d_idx] = 1 - t_box.iou(det.bbox)
    return matrix


def gate_dist_iou_cost(dist_cost:np.ndarray, iou_cost:np.ndarray, \
                        tracks:List[ObjectTrack], detections:List[Detection]) -> Tuple[np.ndarray, np.ndarray]:
    validity_mask = build_task_det_ratio_mask(tracks, detections)
    gated_dist_cost = np.where(validity_mask, dist_cost, INVALID_DIST_DISTANCE)
    gated_iou_cost = np.where(validity_mask, iou_cost, INVALID_IOU_DISTANCE)
    return gated_dist_cost, gated_iou_cost


def build_dist_iou_cost(kf:KalmanFilter, tracks:List[ObjectTrack], detections:List[Detection]) \
    -> Tuple[np.ndarray, np.ndarray]:
    dist_cost = build_dist_cost(kf, tracks, detections)
    iou_cost = build_iou_cost(tracks, detections)
    return gate_dist_iou_cost(dist_cost, iou_cost, tracks, detections)
    
    # bad_ratio_mask = ~build_task_det_ratio_mask(tracks, detections)
    # iou_cost[bad_ratio_mask] = INVALID_IOU_DISTANCE
    # dist_cost[bad_ratio_mask] = INVALID_DIST_DISTANCE

    # return dist_cost, iou_cost


_AREA_RATIO_LIMITS = (0.3, 2.8)
_LARGE_AREA_RATIO_LIMITS = (0.5, 2)
def build_task_det_ratio_mask(tracks:List[ObjectTrack], detections:List[Detection],
                                area_ratio_limits:npt.ArrayLike=_AREA_RATIO_LIMITS):
    det_areas = np.array([det.bbox.area() for det in detections])
    
    area_ratio_limits = np.array(area_ratio_limits)
    large_area_ratio_limits = np.array(area_ratio_limits)

    mask = np.zeros((len(tracks), len(detections)), dtype=bool)
    for t_idx, track in enumerate(tracks):
        t_area = track.location.area()
        ratio_limits = area_ratio_limits if t_area < 100000 else large_area_ratio_limits
        limits = ratio_limits * t_area
        mask[t_idx,:] = (det_areas >= limits[0]) & (det_areas <= limits[1])
        # print(track.location.area(), [area / track.location.area() for area in det_areas])
        
    return mask
    

# def build_metric_cost(tracks:List[ObjectTrack], detections:List[Detection], dist_cost:np.ndarray,
#                         gate_distance:float, track_idxes:List[int], det_idxes:List[int]):
#     metric_cost = build_raw_metric_cost(tracks, detections, track_idxes, det_idxes)
#     gate_metric_cost(metric_cost, dist_cost, tracks, detections, gate_distance)

#     return metric_cost

def build_metric_cost(tracks:List[ObjectTrack], detections:List[Detection],
                        track_idxes:List[int], det_idxes:List[int]) -> np.ndarray:
    def build_matrix(tracks:List[ObjectTrack], features) -> np.ndarray:
        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, track in enumerate(tracks):
            samples = track.features
            if samples and len(features) > 0:
                distances = utils.cosine_distance(samples, features)
                cost_matrix[i, :] = distances.min(axis=0)
        return cost_matrix

    reduced_tracks = list(utils.get_items(tracks, track_idxes))
    reduced_features = list(det.feature for det in utils.get_items(detections, det_idxes))
    reduced_matrix = build_matrix(reduced_tracks, reduced_features)

    cost_matrix = np.ones((len(tracks), len(detections)))
    for row_idx, t_idx in enumerate(track_idxes):
        for col_idx, d_idx in enumerate(det_idxes):
            cost_matrix[t_idx, d_idx] = reduced_matrix[row_idx, col_idx]
    return cost_matrix

def gate_metric_cost(metric_costs:np.ndarray, dist_costs:np.ndarray,
                    gate_threshold:float) -> None:
    gated = np.where(dist_costs > gate_threshold, INVALID_METRIC_DISTANCE, metric_costs)
    return gated
    # for t_idx, track in enumerate(tracks):
    #     t_box = track.location
    #     for d_idx, det in enumerate(detections):
    #         if dist_costs[t_idx, d_idx] == INVALID_DIST_DISTANCE:
    #             metric_costs[t_idx, d_idx] = INVALID_METRIC_DISTANCE
    #         elif dist_costs[t_idx, d_idx] > gate_threshold:
    #             center_dist = t_box.center().distance_to(det.bbox.center())
    #             if center_dist > 150:
    #                 metric_costs[t_idx, d_idx] = INVALID_METRIC_DISTANCE


def print_cost_matrix(tracks:List[ObjectTrack], cost, trim_overflow=None):
    if trim_overflow:
        cost = cost.copy()
        cost[cost > trim_overflow] = trim_overflow

    for tidx, track in enumerate(tracks):
        dists = [int(round(v)) for v in cost[tidx]]
        track_str = f" {tidx:02d}: {track.id:03d}({track.state},{track.time_since_update:02d})"
        dist_str = ', '.join([f"{v:4d}" if v != trim_overflow else "    " for v in dists])
        print(f"{track_str}: {dist_str}")