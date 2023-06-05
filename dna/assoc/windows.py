from __future__ import annotations

from typing import Union, Optional, List, Tuple, Set, Dict, Iterable, Generator

import sys
import logging

from dna.support import iterables
from dna.event import NodeId, EventProcessor, TrackEvent, TimeElapsed, TrackDeleted, TrackletId
from .association import Association
from .collection import AssociationCollector
from .closure import AssociationClosure, AssociationClosureBuilder


class Window:
    __slots__ = ('begin_millis', 'end_millis', 'events')
    
    def __init__(self, begin_millis:int, end_millis:int) -> None:
        self.begin_millis = begin_millis
        self.end_millis = end_millis
        self.events:List[TrackEvent] = []
        
    def range_contains(self, ev:TrackEvent) -> bool:
        return ev.ts >= self.begin_millis and ev.ts < self.end_millis
        
    def append(self, te:TrackEvent) -> None:
        self.events.append(te)
            
    def remove(self, trk_id:TrackletId) -> None:
        """주어진 tracklet id를 갖는 모든 TrackEvent를 삭제한다.

        Args:
            trk_id (TrackletId): 삭제 대상 tracklet의 식별자.
        """
        self.events = [te for te in self.events if te.tracklet_id != trk_id]
        
    def __len__(self) -> int:
        return len(self.events)
        
    def __bool__(self) -> bool:
        return bool(self.events)
    
    def __iter__(self):
        return (ev for ev in self.events)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: time={self.begin_millis}-{self.end_millis}, "
                f"length={len(self.events)}")
        

class TumblingWindowAssigner(EventProcessor):
    __slots__ = ( 'schema', 'max_ts', 'windows', 'dirty', 'logger' )
    
    def __init__(self, nodes:Iterable[NodeId],
                 window_interval_ms:int,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.max_ts:Dict[NodeId,int] = {node_id:0 for node_id in nodes}
        self.window_interval_ms = window_interval_ms
        self.windows:List[Window] = []
        self.dirty = True
        self.logger = logger
        
    def close(self) -> None:
        # 잔여 window에 포함된 TrackEvent에 대해 association을 생성한다.
        for window in self.windows:
            self._publish_event((window, None))
        self.windows.clear()
            
        super().close()
        
    def handle_event(self, ev:Union[TrackEvent,TimeElapsed]) -> None:
        if isinstance(ev, TrackEvent):
            self.dirty = True
            
            # 너무 늦게 도착한 TrackEvent인 경우는 무시한다.
            if ev.node_id not in self.max_ts:
                return
            
            if ev.is_confirmed() or ev.is_tentative():
                if ev.ts > self.max_ts[ev.node_id]:
                    self.max_ts[ev.node_id] = ev.ts
                
                # 본 TrackEvent를 해당 window에 추가하고, close된 window들에 대해 association들을 생성한다.
                self.populate_window(ev)
                for window in self._generate_closed_windows():
                    self._publish_event((window, None))
            elif ev.is_deleted():
                # 대상 TrackEvent 생성 시점보다 빠른 영역을 갖는 모든 window에 대해
                # 종료된 tracklet과 관련된 association을 생성하고, 관련 window들에서
                # 종료된 tracklet에 의해 생성된 TrackEvent를 모두 삭제한다.
                for window in self.windows:
                    if ev.ts < window.begin_millis:
                        break
                    self._publish_event((window, ev.tracklet_id))
                    window.remove(ev.tracklet_id)
                self._publish_event(ev)
        elif isinstance(ev, TimeElapsed):
            # 바로 전 TimeElapsed 이벤트를 받은 후부터 지금까지 한번도 TrackEvent를 받지 않은 경우에는
            # 가장 오래된 window를 통한 association을 생성하고, 해당 window를 제거한다.
            if not self.dirty and self.windows:
                window = self.windows.pop(0)
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"publish idle window: {window}")
                self._publish_event((window, None))
            self.dirty = False
                        
    def populate_window(self, te:TrackEvent) -> None:
        if self.windows:
            if te.ts < self.windows[0].begin_millis:
                if self.logger and self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f"drop too-late event: {te}")
                return
            
            while True:
                # TrackEvent가 포함될 window를 검색한다.
                window = iterables.first((w for w in self.windows if w.range_contains(te)))
                if window is not None:
                    window.append(te)
                    return
                
                # 해당 window가 존재하지 않는 경우는 다음 time slot의 window를 생성하여 추가함.
                last_window = self.windows[-1]
                next_window = Window(begin_millis=last_window.end_millis,
                                     end_millis=last_window.end_millis+self.window_interval_ms)
                self.windows.append(next_window)
        else:
            window = Window(begin_millis=te.ts, end_millis=te.ts + self.window_interval_ms)
            window.append(te)
            self.windows.append(window)
        
    def _generate_closed_windows(self) -> Generator[Window, None, None]:
        min_ts = min(self.max_ts.values(), default=0)
        max_ts = max(self.max_ts.values()) - (2*1000)
        while self.windows:
            first_window = self.windows[0]
            if first_window.end_millis < min_ts:
                self.windows.pop(0)
                yield first_window
            elif first_window.end_millis < max_ts - (2*1000):
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"flush frozen window: {first_window}, max_ts={max_ts}")
                self.windows.pop(0)
                yield first_window
            else:
                return