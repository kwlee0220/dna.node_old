
from __future__ import annotations
from typing import Tuple, List, Optional, Set


from abc import ABCMeta, abstractmethod
class Matcher(metaclass=ABCMeta):
    @abstractmethod
    def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
        -> Tuple[List[Tuple[int,int]], List[int], List[int]]: pass
        
    @staticmethod
    def chain(*matchers) -> Matcher:
        return ChainedMatcher(*matchers)


class ChainedMatcher(Matcher):
    def __init__(self, *matchers):
        self.matchers = matchers
        
    def match(self, row_idxes:Optional[List[int]]=None, column_idxes:Optional[List[int]]=None) \
        -> Tuple[List[Tuple[int,int]], List[int], List[int]]:

        row_idxes = range(self.matrix.shape[0]) if row_idxes is None else row_idxes
        column_idxes = range(self.matrix.shape[1]) if column_idxes is None else column_idxes
        
        unmatched_row_idxes = row_idxes.copy()
        unmatched_col_idxes = column_idxes.copy()
        
        matches = []
        for matcher in self.matchers:
            if not (unmatched_row_idxes and unmatched_col_idxes):
                break
            
            matches0, unmatched_row_idxes, unmatched_col_idxes = matcher.match(unmatched_row_idxes, unmatched_col_idxes)
            if matches0:
                matches += matches0
        return matches, unmatched_row_idxes, unmatched_col_idxes
    
         