from __future__ import annotations
from typing import Optional, List, Tuple, Iterable, Generator
from datetime import timedelta
import logging

import numpy as np

from dna.support import iterables
from dna.node import TrackEvent, TrackDeleted, TrackletId, EventProcessor
from dna.node.event_processors import TimeElapsedGenerator, EventRelay
from .windows import Window, TumblingWindowAssigner
from .association import Association, BinaryAssociation
from .schema import NodeAssociationSchema
        

class MotionBasedTrackletAssociator(EventProcessor):
    def __init__(self,
                 schema:NodeAssociationSchema,
                 window_interval_ms:int,
                 max_distance_to_camera:float,
                 max_track_distance:float,
                 idle_timeout:float=1,
                 *,
                 logger:Optional[logging.Logger]=None) -> None:
        super().__init__()
        
        self.schema = schema
        self.max_distance_to_camera = max_distance_to_camera
        self.logger = logger
        
        self.ticker = TimeElapsedGenerator(interval=timedelta(seconds=idle_timeout))
        
        self.windows = TumblingWindowAssigner(nodes=schema.nodes,
                                              window_interval_ms=window_interval_ms,
                                              logger=logger.getChild('window') if logger else None)
        self.ticker.add_listener(self.windows)
        
        self.associator = MotionBasedAssociator(schema=schema, max_track_distance=max_track_distance)
        self.windows.add_listener(self.associator)
        
        self.associator.add_listener(EventRelay(target=self))
        
    def close(self) -> None:
        self.ticker.stop()
        self.windows.close()
        super().close()
        
    def start(self) -> None:
        self.ticker.start()
        
    def handle_event(self, ev:TrackEvent) -> None:
        # 일정 거리 이상에서 추정된 물체 위치 정보는 사용하지 않는다.
        if (ev.is_confirmed() or ev.is_tentative()) and ev.distance > self.max_distance_to_camera:
            return
        self.windows.handle_event(ev)
            
    def __len__(self) -> int:
        return len(self.collector)
        
    def __iter__(self) -> Iterable[Association]:
        return iter(self.collector)
        
    def find(self, key:Tuple[TrackletId,TrackletId]) -> Optional[Association]:
        closure = self.collector.find(key)
        return self.collector.find(key)

    def get_score(self, trk_pair:Tuple[TrackletId, TrackletId], *, estimate:bool=False) -> Optional[float]:
        closure = self.collector.find(trk_pair[0])
        if closure is None:
            return None
        
        if trk_pair[1] in closure.tracklets():
            assoc = closure.find((trk_pair[0].node_id, trk_pair[1].node_id), estimate=estimate)
            return assoc.score if assoc else None
        else:
            return None


class MotionBasedAssociator(EventProcessor):
    def __init__(self, schema:NodeAssociationSchema, max_track_distance:float) -> None:
        super().__init__()
        
        self.schema = schema
        self.max_track_distance = max_track_distance
        
    def close(self) -> None:
        super().close()
        
    def handle_event(self, ev:Tuple[Window,Optional[TrackletId]]|TrackEvent) -> None:
        if isinstance(ev, TrackEvent):
            if ev.is_deleted():
                self._publish_event(TrackDeleted(node_id=ev.node_id, track_id=ev.track_id, ts=ev.ts))
        else:
            for assoc in self.associate(ev[0].events, tracklet_id=ev[1]):
                self._publish_event(assoc)
        
    def associate(self, track_events:List[TrackEvent], *,
                  tracklet_id:Optional[TrackletId]=None) -> Generator[Association, None, None]: 
        def calc_split_distance(events1:List[TrackEvent], events2:List[TrackEvent]) -> float:
            dist = np.mean([te1.world_coord.distance_to(te2.world_coord)
                            for te1, te2 in zip(events1, events2)])
            return dist
        def calc_distance(ev_list1:List[TrackEvent], ev_list2:List[TrackEvent]) -> Association:
            if len(ev_list1) >= len(ev_list2):
                splits = iterables.buffer_iterable(ev_list1, len(ev_list2), skip=1)
                base = ev_list2
            else:
                splits = iterables.buffer_iterable(ev_list2, len(ev_list1), skip=1)
                base = ev_list1
            return min(calc_split_distance(base, split) for split in splits)
        
        # 주어진 TrackEvent들을 tracklet별로 그룹핑한다.
        tracklets = iterables.groupby(track_events, key=lambda v: v.tracklet_id)
        
        if tracklet_id and tracklet_id not in tracklets:
            return
        
        t1 = tracklet_id if tracklet_id else iterables.first(tracklets.keys())
        while tracklets:
            events1 = tracklets.pop(t1)
            for peer_node in self.schema.peers(t1.node_id):
                peer_tracklet_ids = (t2 for t2 in tracklets.keys() if t2.node_id == peer_node)
                for t2 in peer_tracklet_ids:
                    events2 = tracklets[t2]
                
                    dist = calc_distance(events1, events2)
                    if dist <= self.max_track_distance:
                        score = 1 - (dist / self.max_track_distance)
                        ts = max(events1[-1].ts, events2[-1].ts)
                        yield BinaryAssociation(t1, t2, score, ts)
                        
            if tracklet_id:
                return
            t1 = iterables.first(tracklets.keys())