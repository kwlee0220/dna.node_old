
from __future__ import annotations
from typing import Tuple, Iterable, List, Optional, Callable

import logging
import numpy as np
import numpy.typing as npt

from dna.detect import Detection
from .base import Matcher
from .. import utils


class ReciprocalCostMatcher(Matcher):
    def __init__(self, cost_matrix:np.ndarray, threshold:float, weights:npt.ArrayLike, logger:Optional[logging.Logger]=None) -> None:
        self.cost_matrix = cost_matrix
        self.threshold = threshold
        self.weights = np.array(weights)
        self.logger = logger
        
    def match(self, row_idxes:Iterable[int], column_idxes:Iterable[int]) -> List[Tuple[int,int]]:
        unmatched_row_idxes = row_idxes.copy()
        unmatched_col_idxes = column_idxes.copy()
        
        matches = []
        match_count = 1
        while match_count > 0 and (unmatched_row_idxes and unmatched_col_idxes):
            match_count = 0
            row_idxes = unmatched_row_idxes.copy()
            for row_idx in row_idxes:
                top_col_idx = self.select_reciprocal_top(self.threshold, row_idx, unmatched_row_idxes, unmatched_col_idxes)
                if top_col_idx is not None:
                    matches.append((row_idx, top_col_idx))
                    unmatched_row_idxes.remove(row_idx)
                    unmatched_col_idxes.remove(top_col_idx)
                    match_count += 1
                    if not (unmatched_row_idxes and unmatched_col_idxes):
                        break
                        
        return matches

    def select_reciprocal_top(self, threshold:float, row_idx:int, row_idxes:Iterable[int], col_idxes:Iterable[int]) -> int:
        row = self.cost_matrix[row_idx, :]
        top1_col_idx, top2_col_idx = self.select_top2(row, threshold, col_idxes, self.weights)
        if top1_col_idx is None:    # (None, None)
            return None
        elif top2_col_idx is None:  # (top1, None)
            return top1_col_idx if self.is_reciprocal_top(row_idx, top1_col_idx, row_idxes) else None
        else:                       # (top1, top2)
            if row[top1_col_idx] * 2 < row[top2_col_idx]:
                # 독보적으로 작은 값이 존재하는 경우
                return top1_col_idx if self.is_reciprocal_top(row_idx, top1_col_idx, row_idxes) else None
            else:
                # 독보적으로 작은 값이 없는 경우
                if self.is_reciprocal_top(row_idx, top1_col_idx, row_idxes) \
                    and self.is_reciprocal_top(row_idx, top2_col_idx, row_idxes):
                    return top1_col_idx
                else:
                    return None

    def is_reciprocal_top(self, row_idx:int, col_idx:int, row_idxes:List[int]):
        column = self.cost_matrix[:, col_idx]

        top_idx1, top_idx2 = self.select_top2(column, self.threshold, row_idxes)
        if row_idx != top_idx1:
            return False

        if top_idx2 is None:
            return True
        else:
            closest1, closest2 = column[top_idx1], column[top_idx2]
            return closest1 * 2 < closest2

    def select_top2(self, values:np.ndarray, threshold:float, idxes:Iterable[int], weights=1) -> Tuple[int,int]:
        selected_idxes = np.where(values <= threshold)[0]
        selected_idxes = utils.intersection(selected_idxes, idxes)

        n_selecteds = len(selected_idxes)
        if n_selecteds == 0:
            return (None, None)
        if n_selecteds == 1:
            return (selected_idxes[0], None)
        elif n_selecteds == 2:
            weighted_values = values * weights
            first, second = weighted_values[selected_idxes[0]], weighted_values[selected_idxes[1]]
            return (selected_idxes[0], selected_idxes[1]) if first <= second else (selected_idxes[1], selected_idxes[0])
        else:
            weighted_values = values * weights
            selecteds = weighted_values[selected_idxes]
            idxes = np.argpartition(selecteds, 2)
            return (selected_idxes[idxes[0]], selected_idxes[idxes[1]])