from __future__ import annotations
from typing import Any, Iterable, Tuple, List, Dict, Set, Optional, Callable

import sys
import logging

from dna.support import iterables
from dna.node import EventProcessor, EventQueue, EventListener, TrackDeleted
from .association import Association, BinaryAssociation
from .collection import AssociationCollection, AssociationCollector
from .closure import AssociationClosure, AssociationClosureBuilder, PartialMatch
from .associator_motion import NodeAssociationSchema


class AssociationFilter(EventProcessor):
    __slots__ = ('filter', )
    
    def __init__(self, filter:Callable[[Association],bool]) -> None:
        super().__init__()
        self.filter = filter
    
    def handle_event(self, ev:Association|TrackDeleted) -> None:
        if isinstance(ev, Association):
            if self.filter(ev):
                self._publish_event(ev)
        else:
            self._publish_event(ev)
            
            
class KeepEventType(EventProcessor):
    __slots__ = ('types', )
    
    def __init__(self, *types:List[type]) -> None:
        super().__init__()
        self.types = types
    
    def handle_event(self, ev:object) -> None:
        for ev_type in self.types:
            if isinstance(ev, ev_type):
                self._publish_event(ev)
                return
            
            
class ClosedAssociationPublisher(EventProcessor):
    def __init__(self, collection:AssociationCollector,
                 *,
                 publish_best_only:bool=True) -> None:
        super().__init__()
        self.collection = collection
        self.publish_best_only = publish_best_only
        
    def handle_event(self, ev:TrackDeleted|Any) -> None:
        if isinstance(ev, TrackDeleted):
            trk_id = ev.tracklet_id
            newly_closeds = [assoc for assoc in self.collection if trk_id in assoc and assoc.is_closed()]
            if newly_closeds:
                if self.publish_best_only:
                    max_assoc = max(newly_closeds, key=lambda a: a.score)
                    self._publish_event(max_assoc)
                else:
                    for assoc in newly_closeds:
                        self._publish_event(assoc)
                        

class AssociationCloser(EventListener):
    def __init__(self, collection:AssociationCollection) -> None:
        super().__init__()
        self.collection = collection
        
    def handle_event(self, track_deleted:TrackDeleted) -> None:
        if isinstance(track_deleted, TrackDeleted):
            trk = track_deleted.tracklet_id
            for assoc in self.collection.query(condition=PartialMatch(trk)):
                assoc.close(tracklet=trk)
                

class FixedIntervalClosureBuilder(EventProcessor):
    __slots__ = ( 'interval_ms', 'first_ms', 'last_ms', 'collector', 'publish_best_only', 'logger' )
    
    def __init__(self, interval_ms:int,
                 *,
                 closure_collector:Optional[AssociationClosureBuilder]=None,
                 publish_best_only:bool=False,
                 token:Optional[str]=None,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.interval_ms = interval_ms
        self.first_ms:int = sys.maxsize
        self.last_ms:int = 0
        self.closure_collector = closure_collector if closure_collector else AssociationClosureBuilder()
        self.publish_best_only = publish_best_only
        self.token = token
        self.logger = logger
        
    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, Association):
            # 추가된 association을 바로 closure 생성에 사용하지 않고, 'interval_ms' 동안 수집한다.
            self.closure_collector.handle_event(ev)
            self.first_ms = min(self.first_ms, ev.ts)
            self.last_ms = max(self.last_ms, ev.ts)
            
            if (self.last_ms - self.first_ms) > self.interval_ms:
                closures = [closure for closure in self.closure_collector.closures if closure.ts >= self.first_ms]
                if closures:
                    self.__publish_closures(closures)
                self.first_ms = self.last_ms
        
    def __publish_closures(self, closures:List[AssociationClosure]):
        if self.publish_best_only:
            closures.sort(key=lambda a: a.score, reverse=True)
            while closures:
                top = closures[0].copy()
                closures = [c for c in closures[1:] if c.is_disjoint(top)]
                self._publish_event(top)
        else:
            for closure in closures:
                self._publish_event(closure)


class JoinInput(EventListener):
    def __init__(self, handler, index:int) -> None:
        super().__init__()
        self.handler = handler
        self.index = index
        
    def handle_event(self, ev):
        self.handler.handle_event((ev, self.index))
        

class LeftOuterJoin(EventProcessor):
    def __init__(self, left:AssociationCollection, *, merge:bool=True) -> None:
        super().__init__()
        self.left = left
        self.merge = merge
    
    def handle_event(self, ev:AssociationClosure) -> None:
        if isinstance(ev, AssociationClosure):
            if any(left_assoc.is_conflict(ev) for left_assoc in self.left):
                return
            
            for left_assoc in self.left:
                if left_assoc.is_mergeable(ev):
                    if self.merge:
                        match:AssociationClosure = left_assoc.copy()
                        for left_assoc in ev.source_associations:
                            match.extend(left_assoc)
                        self._publish_event(match)
                    else:
                        self._publish_event(ev)
                    return
        

class ExtendLeft(EventProcessor):
    def __init__(self, left:AssociationCollection) -> None:
        super().__init__()
        self.left = left
    
    def handle_event(self, ev:AssociationClosure) -> None:
        if isinstance(ev, AssociationClosure):
            if any(left_assoc.is_conflict(ev) for left_assoc in self.left):
                return
            
            for left_assoc in self.left:
                if left_assoc.is_mergeable(ev):
                    old_score = left_assoc.score
                    for src_assoc in ev.source_associations:
                        left_assoc.add(src_assoc)
                    if old_score != left_assoc.score:
                        self._publish_event(left_assoc)
        
        
class WeightedSumJoin(EventProcessor):
    def __init__(self, schema:NodeAssociationSchema, source0:EventQueue, source1:EventQueue, weight:float) -> None:
        super().__init__()
        
        self.schema = schema
        self.weight = weight
        
        collector0 = AssociationCollector(publish_on_update=True, delete_on_finished=False)
        source0.add_listener(collector0)
        input0 = JoinInput(self, 0)
        collector0.add_listener(input0)
        
        collector1 = AssociationCollector(publish_on_update=True, delete_on_finished=False)
        source1.add_listener(collector1)
        input1 = JoinInput(self, 1)
        collector1.add_listener(input1)
        
        self.collectors = collector0, collector1
        self.inputs = input0, input1
    
    def handle_event(self, ev:Tuple[Association|TrackDeleted,int]) -> None:
        if isinstance(ev[0], Association):
            sum_assoc = self.join(ev[0], ev[1])
            if sum_assoc:
                self._publish_event(sum)
        elif isinstance(ev[0], TrackDeleted):
            pass
        
    def join(self, target:Association, index:int) -> Optional[Association]:
        self._publish_event(target)
        # if index == 0:
        #     assoc = self.collectors[1].find(target.key())
        #     if assoc:
        #         sum = target.score * self.weight + assoc.score * (1-self.weight)
        #         return Association(target.tracklet1, target.tracklet2, sum)
        # elif index == 1:
        #     assoc = self.collectors[0].find(target.key())
        #     if assoc:
        #         sum = assoc.score * self.weight + target.score * (1-self.weight)
        #         return Association(target.tracklet1, target.tracklet2, sum)
            
        return None