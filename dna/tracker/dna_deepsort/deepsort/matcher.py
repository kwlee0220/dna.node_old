from __future__ import annotations
from typing import Tuple, List, Optional, Set
import sys

import numpy as np
from numpy.linalg import det
from scipy.optimize import linear_sum_assignment

import dna
from dna import Box
from dna.detect import Detection
from dna.tracker.matcher import Matcher, ChainedMatcher
from . import utils
from .track import Track

import logging
LOGGER = logging.getLogger('dna.tracker.deepsort')


_HUGE = 300
_LARGE = 150
_MEDIUM = 27
_COMBINED_METRIC_THRESHOLD_4S = 0.89
_COMBINED_METRIC_THRESHOLD_4M = 0.55
_COMBINED_METRIC_THRESHOLD_4L = 0.40
_COMBINED_DIST_THRESHOLD_4S = 75
_COMBINED_DIST_THRESHOLD_4M = 150
_COMBINED_DIST_THRESHOLD_4L = 310
_COMBINED_INFINITE = 9.99

def create_matrix(matrix, threshold, mask=None):
    new_matrix = matrix.copy()
    if mask is not None:
        new_matrix[mask] = threshold + 0.00001
    new_matrix[new_matrix > threshold] = threshold + 0.00001
    return new_matrix

def hot_unconfirmed_mask(cmatrix, threshold, track_indices, detection_indices):
    mask = np.zeros(cmatrix.shape, dtype=bool)
    
    target_dets = set()
    for tidx in track_indices:
        selecteds = np.where(cmatrix[tidx,:] < threshold)[0]
        target_dets = target_dets.union(selecteds)
    for didx in detection_indices:
        mask[:,didx] = cmatrix[:,didx] >= 0.1

    return mask

def size_class(tlwh):
    if tlwh[2] >= _LARGE and tlwh[3] >= _LARGE: # large detections
        return 'L'
    elif tlwh[2] >= _MEDIUM and tlwh[3] >= _MEDIUM: # medium detections
        return 'M'
    else: # small detections
        return 'S'

import math
def gate_metric_cost(metric_costs, dist_costs, tracks:List[Track], detections:List[Detection]):
    mask = dist_costs > 500
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            if mask[t_idx, d_idx]:
                box_dist = utils.track_to_box(tracks[t_idx]).min_distance_to(detections[d_idx].bbox)
                mask[t_idx, d_idx] = box_dist > 200
    copied_metric = metric_costs.copy()
    copied_metric[mask] = 1
    return copied_metric

def print_matrix(tracks, detections, matrix, threshold, track_indice, detection_indices):
    def pattern(v):
        return "    " if v > threshold else f"{v:.2f}"

    col_exprs = []
    for didx, det in enumerate(detections):
        if didx in detection_indices:
            col_exprs.append(f"{didx:-2d}({size_class(det.bbox.to_tlwh())})")
        else:
            col_exprs.append("-----")
    print("              ", ",".join(col_exprs))

    for tidx, track in enumerate(tracks):
        track_str = f"{tidx:02d}: {track.track_id:03d}({track.state},{track.time_since_update:02d})"
        dist_str = ', '.join([pattern(v) for v in matrix[tidx]])
        tag = '*' if tidx in track_indice else ' '
        print(f"{tag}{track_str}: {dist_str}")
        
#########################################################################################
def topk_indices(row, k, col_indices) -> List[int]:
    selecteds = row[col_indices]
    if k == 1:
        return [col_indices[0]]
    elif k >= len(selecteds):
        idxes = np.argsort(selecteds)
        return np.array(col_indices)[idxes].tolist()
    else:
        idxes = np.argpartition(selecteds, k)
        idxes = idxes[:k]
        return np.array(col_indices)[idxes].tolist()

def select_top2_le_threshold_indices(row, threshold, col_indices) -> List[int]:
    selecteds = row[col_indices]
    idxes = np.where(selecteds <= threshold)[0].tolist()
    
    n_selecteds = len(idxes)
    if n_selecteds == 0: 
        return []
    elif n_selecteds == 1:
        return [col_indices[idxes[0]]]
    else:
        idxes = np.array(col_indices)[idxes]
        return topk_indices(row, 2, idxes)

def select_excl_closest(row, candidates:List[int]):
    closest_idx = candidates[0]
    if len(candidates) == 1:
        return closest_idx
    else:
        closest, closest2 = row[closest_idx], row[candidates[1]]
        return closest_idx if closest * 2 < closest2 else None

def match_closest_dist_cover(dist_matrix, threshold:float, exclusive:bool, track_indices:List[int], det_indices:List[int]) \
    -> Tuple[
        List[Tuple[int,int]],
        List[int],
        List[int],
        List[int]
    ]:
    matches = []
    unmatched_tracks = track_indices
    unmatched_detections = det_indices
    while True:
        matches0, unmatched_tracks, unmatched_detections \
            = match_closest_dist(dist_matrix, threshold, exclusive, unmatched_tracks, unmatched_detections)
        matches += matches0
        if not (matches0 and unmatched_tracks and unmatched_detections):
            return matches, unmatched_tracks, unmatched_detections
    
def match_closest_dist(dist_matrix, threshold:float, exclusive:bool, track_indices:List[int], det_indices:List[int]) \
    -> Tuple[
        List[Tuple[int,int]],
        List[int],
        List[int]
    ]:
    matches = []
    unmatched_tracks = track_indices.copy()
    unmatched_detections = det_indices.copy()

    for tidx in track_indices:
        det_dists = dist_matrix[tidx, :]
        close_det_idxes = select_top2_le_threshold_indices(det_dists, threshold, det_indices)
        if close_det_idxes:
            closest_didx = select_excl_closest(det_dists, close_det_idxes) if exclusive else close_det_idxes[0]
            if closest_didx is not None:
                # 가장 가까운 detection이 두번째로 가까운 detection보다 월등히 가까운 경우
                # 가장 가까운 detection을 기준으로 남은 track과의 거리를 구해서 
                # 'tidx' track이 다른 track들에 비해 월등히 가까운지 확인함.
                track_dists = dist_matrix[:, closest_didx]
                close_tidxes = select_top2_le_threshold_indices(track_dists, threshold, track_indices)
                closest_tidx = select_excl_closest(track_dists, close_tidxes)
                if closest_tidx == tidx:
                    # 가장 가까운 detection을 기준으로 월등히 가장 가까운 track이 바로 'tidx'인 경우
                    # 이 두 track과 detection은 다른 어떤 track-detection 조합보다 가깝기 때문에
                    # 이 둘을 match 시킨다.
                    matches.append((tidx, closest_didx))
                    unmatched_tracks.remove(tidx)
                    unmatched_detections.remove(closest_didx)
                    if len(unmatched_detections) == 0:
                        break
                    
    return matches, unmatched_tracks, unmatched_detections
#########################################################################################

def match_iou_cost(tracks, detections, threshold, track_indices, det_indices) \
    -> Tuple[
        List[Tuple[int,int]],
        List[int],
        List[int]
    ]:
    matrix = iou_cost_matrix(tracks, detections, track_indices, det_indices)
    return matching_by_hungarian(matrix, threshold, track_indices, det_indices)

def matching_by_hungarian(cost_matrix, threshold, row_idxes, col_idxes):
    def _remove_by_index(list, idx):
        remains = list.copy()
        removed = remains.pop(idx)
        return removed, remains

    n_rows, n_cols = len(row_idxes), len(col_idxes)
    if n_rows == 0 or n_cols == 0:
        return [], row_idxes, col_idxes

    if n_rows == 1 and n_cols == 1:
        if cost_matrix[row_idxes[0], col_idxes[0]] <= threshold:
            return [(row_idxes[0], col_idxes[0])], [], []
        else:
            return [], row_idxes, col_idxes
    elif n_cols == 1:   # track만 여러개
        reduced = cost_matrix[:,col_idxes[0]][row_idxes]
        tidx = np.argmin(reduced)
        if reduced[tidx] <= threshold:
            matched_track, unmatched_tracks = _remove_by_index(row_idxes, tidx)
            return [(matched_track, col_idxes[0])], unmatched_tracks, []
        else:
            return [], row_idxes, col_idxes
    elif n_rows == 1:       # detection만 여러개
        reduced = cost_matrix[row_idxes[0],:][col_idxes]
        didx = np.argmin(reduced)
        if reduced[didx] <= threshold:
            matched_det, unmatched_dets = _remove_by_index(col_idxes, didx)
            return [(row_idxes[0], matched_det)], [], unmatched_dets
        else:
            return [], row_idxes, col_idxes

    matrix = cost_matrix[np.ix_(row_idxes, col_idxes)]
    indices = linear_sum_assignment(matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    matches = []
    unmatched_tracks = row_idxes.copy()
    unmatched_detections = col_idxes.copy()
    for i, j in indices:
        tidx = row_idxes[i]
        didx = col_idxes[j]
        if cost_matrix[tidx, didx] <= threshold:
            matches.append((tidx, didx))
            unmatched_tracks.remove(tidx)
            unmatched_detections.remove(didx)
    
    return matches, unmatched_tracks, unmatched_detections

def overlap_detections(target_indices:List[int], detection_indices:List[int], detections:List[Detection], threshold:float):
    boxes = [det.bbox for det in detections]

    overlaps = []
    for target_idx in target_indices:
        target = boxes[target_idx]
        for idx, ov in utils.overlap_boxes(target, boxes, detection_indices):
            if max(ov) >= threshold:
                overlaps.append((idx, ov[0], ov[1]))
    return overlaps

def delete_overlapped_tentative_tracks(tracks, threshold):
    confirmed_track_idxs = [i for i, t in enumerate(tracks) 
                                if t.is_confirmed() and t.time_since_update == 1 and not t.is_deleted()]
    unconfirmed_track_idxs = [i for i, t in enumerate(tracks)
                                if not t.is_confirmed() and not t.is_deleted()]
    if not (confirmed_track_idxs and unconfirmed_track_idxs):
        return

    t_boxes = [utils.track_to_box(track) for track in tracks]
    for uc_idx in unconfirmed_track_idxs:
        for t_idx, ov in utils.overlap_boxes(t_boxes[uc_idx], t_boxes, confirmed_track_idxs):
            if max(ov) >= threshold:
                tracks[uc_idx].mark_deleted()
                if LOGGER.isEnabledFor(logging.DEBUG):
                    uc_track_id = tracks[uc_idx].track_id
                    ov_track_id = tracks[t_idx].track_id
                    ov_ratio = max(ov[1])
                    LOGGER.debug((f"delete tentative track[{uc_track_id}] because it is too close to track[{ov_track_id}], "
                                    f"ratio={ov_ratio:.2f}, frame={dna.DEBUG_FRAME_IDX}"))
                    break

def overlap_cost(tracks, detections, track_indices, detention_indices):
    det_boxes = [d.bbox for d in detections]
    ovr_matrix = np.zeros((len(tracks), len(detections)))
    for tidx in track_indices:
        t_box = utils.track_to_box(tracks[tidx])
        for didx in detention_indices:
            ovr_matrix[tidx, didx] = max(t_box.overlap_ratios(det_boxes[didx]))

    return ovr_matrix
    
def iou_cost_matrix(tracks, detections) -> np.array:
    d_boxes = [det.bbox for det in detections]
    
    matrix = np.zeros((len(tracks), len(detections)))
    for t_idx, track in enumerate(tracks):
        t_box = utils.track_to_box(track)
        for d_idx, det in enumerate(detections):
            matrix[t_idx,d_idx] = 1 - t_box.overlap_ratios(d_boxes[d_idx])[2]
    return matrix

def track_det_gate(tracks:List[Track], detections:List[Detection], size_ratio=np.array([0.3,2.8])):
    det_areas = np.array([det.bbox.area() for det in detections])
    gate = np.zeros((len(tracks), len(detections)), dtype=bool)
    for t_idx, track in enumerate(tracks):
        t_area = utils.track_to_box(track).area()
        range = size_ratio * t_area
        gate[t_idx,:] = (det_areas >= range[0]) & (det_areas <= range[1])
        
    return gate

def overlap_cost_matrix(tracks:List[Track], detections:List[Detection], ov_idx:int,
                        track_indices:List[int], detention_indices:List[int]):
    iou_matrix = np.ones((len(tracks), len(detections)))
    for tidx in track_indices:
        t_box = utils.track_to_box(tracks[tidx])
        for didx in detention_indices:
            ovs = t_box.overlap_ratios(detections[didx].bbox)
            print(ovs, np.mean(ovs))
            iou_matrix[tidx, didx] = 1 - np.mean(ovs)

    return iou_matrix


def match_dist_iou_costs(dist_cost, iou_cost, dist_threshold, iou_threshold, exclusive, track_idxes, det_idxes):
    matches, unmatched_track_idxes, unmatched_det_idxes \
        = match_closest_dist_cover(dist_cost, dist_threshold, True, track_idxes, det_idxes)
    if not(unmatched_track_idxes and unmatched_det_idxes):
        return matches, unmatched_track_idxes, unmatched_det_idxes
    
    matches0, unmatched_track_idxes, unmatched_det_idxes \
        = match_closest_dist_cover(iou_cost, iou_threshold, exclusive, unmatched_track_idxes, unmatched_det_idxes)
    matches += matches0
    return matches, unmatched_track_idxes, unmatched_det_idxes




# from abc import ABCMeta, abstractmethod
# class Matcher(metaclass=ABCMeta):
#     @abstractmethod
#     def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
#         -> Tuple[List[Tuple[int,int]], List[int], List[int]]: pass
        
#     @staticmethod
#     def chain(*matchers) -> Matcher:
#         return ChainedMatcher(*matchers)

#     @staticmethod
#     def update_matches(matches:List[Tuple[int,int]], unmatched_row_idxes:List[int], unmatched_col_idxes:List[int],
#                        matches0:List[Tuple[int,int]]) \
#         -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
#         return ( matches+matches0,
#                 utils.subtract(unmatched_row_idxes, utils.project(matches0, 0)),
#                 utils.subtract(unmatched_col_idxes, utils.project(matches0, 1)) )


# class ChainedMatcher(Matcher):
#     def __init__(self, *matchers):
#         self.matchers = matchers
        
#     def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
#         -> Tuple[List[Tuple[int,int]], List[int], List[int]]:

#         row_idxes = range(self.matrix.shape[0]) if row_idxes is None else row_idxes
#         column_idxes = range(self.matrix.shape[1]) if column_idxes is None else column_idxes
        
#         unmatched_row_idxes = row_idxes.copy()
#         unmatched_col_idxes = column_idxes.copy()
        
#         matches = []
#         for matcher in self.matchers:
#             if not (unmatched_row_idxes and unmatched_col_idxes):
#                 break
            
#             matches0, unmatched_row_idxes, unmatched_col_idxes = matcher.match(unmatched_row_idxes, unmatched_col_idxes)
#             if matches0:
#                 matches += matches0
#         return matches, unmatched_row_idxes, unmatched_col_idxes


class HungarianMatcher(Matcher):
    def __init__(self, matrix, threshold:float) -> None:
        self.matrix = matrix
        self.threshold = threshold
        
    def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
        -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        return matching_by_hungarian(self.matrix, self.threshold, row_idxes, column_idxes)
        
    def match_threshold(self, threshold:float, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
        -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        return matching_by_hungarian(self.matrix, threshold, row_idxes, column_idxes)


class ReciprocalCostMatcher(Matcher):
    def __init__(self, matrix, threshold:float, gate_matrix=None, gate_threshold:Optional[float]=None) -> None:
        if gate_matrix is not None and gate_threshold is not None:
            matrix = matrix.copy()
            matrix[gate_matrix > gate_threshold] = threshold + 0.1
        self.matrix = matrix
        self.threshold = threshold
        
    def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
        -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        unmatched_row_idxes = list(range(self.matrix.shape[0])) if row_idxes is None else row_idxes.copy()
        unmatched_col_idxes = list(range(self.matrix.shape[1])) if column_idxes is None else column_idxes.copy()
        
        matches = []
        match_count = 1
        while match_count > 0 and unmatched_col_idxes:
            match_count = 0
            for row_idx in unmatched_row_idxes.copy():
                top_col_idx = self.select_reciprocal_top(self.threshold, row_idx, unmatched_row_idxes, unmatched_col_idxes)
                if top_col_idx is not None:
                    matches.append((row_idx, top_col_idx))
                    unmatched_row_idxes.remove(row_idx)
                    unmatched_col_idxes.remove(top_col_idx)
                    match_count += 1
                    if len(unmatched_col_idxes) == 0:
                        break
                        
        return matches, unmatched_row_idxes, unmatched_col_idxes

    def select_reciprocal_top(self, threshold, row_idx, row_idxes, col_idxes):
        row = self.matrix[row_idx, :]
        top1_col_idx, top2_col_idx = self.select_top2(row, threshold, col_idxes)
        if top1_col_idx is None:    # (None, None)
            return None
        elif top2_col_idx is None:  # (top1, None)
            return top1_col_idx if self.is_reciprocal_top(row_idx, top1_col_idx, row_idxes) else None
        else:                       # (top1, top2)
            if self.is_superior(row, top1_col_idx, top2_col_idx):
                # 독보적으로 작은 값이 존재하는 경우
                return top1_col_idx if self.is_reciprocal_top(row_idx, top1_col_idx, row_idxes) else None
            else:
                # 독보적으로 작은 값이 없는 경우
                if self.is_reciprocal_top(row_idx, top1_col_idx, row_idxes) \
                    and self.is_reciprocal_top(row_idx, top2_col_idx, row_idxes):
                    return top1_col_idx
        return None

    def select_top2(self, values, threshold, idxes) -> Tuple[int,int]:
        selected_idxes = np.where(values <= threshold)[0]
        selected_idxes = utils.intersection(selected_idxes, idxes)
        
        n_selecteds = len(selected_idxes)
        if n_selecteds == 0:
            return (None, None)
        if n_selecteds == 1:
            return (selected_idxes[0], None)
        elif n_selecteds == 2:
            first, second = values[selected_idxes[0]], values[selected_idxes[1]]
            return (selected_idxes[0], selected_idxes[1]) if first <= second else (selected_idxes[1], selected_idxes[0])
        else:
            selecteds = values[selected_idxes]
            idxes = np.argpartition(selecteds, 2)
            return (selected_idxes[idxes[0]], selected_idxes[idxes[1]])

    def is_superior(self, row, top1_idx:int, top2_idx:int) -> int:
        return row[top1_idx] * 2 < row[top2_idx]

    def is_reciprocal_top(self, row_idx, col_idx, row_idxes):
        column = self.matrix[:, col_idx]

        top_idx1, top_idx2 = self.select_top2(column, self.threshold, row_idxes)
        if row_idx != top_idx1:
            return False

        if top_idx2 is None:
            return True
        else:
            closest1, closest2 = column[top_idx1], column[top_idx2]
            return closest1 * 2 < closest2


class CostMatrixBasedMatcher(Matcher):
    def __init__(self, matrix, threshold:float, exclusive:bool=False,
                    fast_match_threshold=0) -> None:
        self.matrix = matrix
        self.threshold = threshold
        self.exclusive = exclusive
        self.fast_match_threshold = fast_match_threshold
        
    def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
        -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        unmatched_row_idxes = list(range(self.matrix.shape[0])) if row_idxes is None else row_idxes.copy()
        unmatched_col_idxes = list(range(self.matrix.shape[1])) if column_idxes is None else column_idxes.copy()
        
        matches = []
        conflict = True
        match_count = 1
        while conflict and match_count > 0 and unmatched_col_idxes:
            conflict = False
            match_count = 0
            
            row_idxes = unmatched_row_idxes.copy()
            for row_idx in row_idxes:
                row = self.matrix[row_idx, :]
                close_col_idxes = self.select_top2(row, self.threshold, unmatched_col_idxes)
                if close_col_idxes:
                    if row[close_col_idxes[0]] < self.fast_match_threshold:
                        matches.append((row_idx, close_col_idxes[0]))
                        unmatched_row_idxes.remove(row_idx)
                        unmatched_col_idxes.remove(close_col_idxes[0])
                        match_count += 1
                        if len(unmatched_col_idxes) == 0:
                            break
                    else:
                        closest_col_idx = self.select_excl_closest(row, close_col_idxes) if self.exclusive else close_col_idxes[0]
                        if closest_col_idx is not None:
                            # 가장 가까운 detection이 두번째로 가까운 detection보다 월등히 가까운 경우
                            # 가장 가까운 detection을 기준으로 남은 track과의 거리를 구해서 
                            # 'tidx' track이 다른 track들에 비해 월등히 가까운지 확인함.
                            column = self.matrix[:, closest_col_idx]
                            close_row_idxes = self.select_top2_le_threshold_indices(column, self.threshold, unmatched_row_idxes)
                            closest_row_idx = self.select_excl_closest(column, close_row_idxes)
                            if closest_row_idx == row_idx:
                                # 가장 가까운 detection을 기준으로 월등히 가장 가까운 track이 바로 'tidx'인 경우
                                # 이 두 track과 detection은 다른 어떤 track-detection 조합보다 가깝기 때문에
                                # 이 둘을 match 시킨다.
                                matches.append((row_idx, closest_col_idx))
                                unmatched_row_idxes.remove(row_idx)
                                unmatched_col_idxes.remove(closest_col_idx)
                                match_count += 1
                                if len(unmatched_col_idxes) == 0:
                                    break
                            else:
                                conflict = True
                        else:
                            conflict = True
                        
        return matches, unmatched_row_idxes, unmatched_col_idxes
    
    def select_top2(self, values, threshold, idxes) -> List[int]:
        selected_idxes = np.where(values <= threshold)[0]
        selected_idxes = utils.intersection(selected_idxes, idxes)
        
        n_selecteds = len(selected_idxes)
        if n_selecteds == 0: 
            return []
        elif n_selecteds == 1:
            return selected_idxes
        elif n_selecteds == 2:
            first, second = values[selected_idxes[0]], values[selected_idxes[1]]
            if first <= second:
                return selected_idxes
            else:
                return [selected_idxes[1], selected_idxes[0]]
        else:
            selecteds = values[selected_idxes]
            idxes = np.argpartition(selecteds, 2)
            return [selected_idxes[idxes[0]], selected_idxes[idxes[1]]]
    
    def select_top2_le_threshold_indices(self, values, threshold, idxes) -> List[int]:
        selected_idxes = np.where(values <= threshold)[0]
        selected_idxes = utils.intersection(selected_idxes, idxes)
        
        n_selecteds = len(selected_idxes)
        if n_selecteds == 0: 
            return []
        elif n_selecteds == 1:
            return selected_idxes
        else:
            return topk_indices(values, 2, selected_idxes)

    def select_excl_closest(self, row, idx_ranks:List[int]):
        closest_idx = idx_ranks[0]
        if len(idx_ranks) == 1:
            return closest_idx
        else:
            closest, closest2 = row[closest_idx], row[idx_ranks[1]]
            return closest_idx if closest * 2 < closest2 else None


def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)