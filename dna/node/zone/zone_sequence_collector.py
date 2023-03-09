from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional, Union

import logging
from datetime import timedelta

from ..event_processor import EventProcessor
from dna.node.zone import ZoneEvent, TrackDeleted, ZoneVisit, ZoneSequence

LOGGER = logging.getLogger('dna.node.zone.Turn')


class ZoneSequenceCollector(EventProcessor):
    def __init__(self) -> None:
        super().__init__()
        
        self.sequences:Dict[int,ZoneSequence] = dict()
    
    def close(self) -> None:
        self.sequences.clear()
        super().close()

    def handle_event(self, ev:Union[ZoneEvent,TrackDeleted]) -> None:
        if isinstance(ev, ZoneEvent):
            self.handle_zone_event(ev)
        elif isinstance(ev, TrackDeleted):
            self.handle_track_deleted(ev)
            
    def handle_zone_event(self, ev:ZoneEvent) -> None:
        if ev.is_inside() or ev.is_unassigned():
            return
        
        seq = self.sequences.get(ev.track_id)
        if seq is None:
            seq = ZoneSequence(track_id=ev.track_id, visits=[])
            self.sequences[ev.track_id] = seq
            
        if ev.is_entered():
            seq.append(ZoneVisit.open(ev))
        elif ev.is_left():
            last:ZoneVisit = seq[-1]
            assert last.is_open()
            last.close(frame_index=ev.frame_index, ts=ev.ts)
        elif ev.is_through():
            # 현재 특정 zone 안에 있는 경우는 해당 zone에서 leave하는 것을 추가함
            last = seq[-1] if len(seq) > 0 else None
            if last and last.zone_id != ev.zone_id:
                last.close(frame_index=ev.frame_index, ts=ev.ts)
                self.publish_event(seq.duplicate())
                last = None

            if last is None:
                last = ZoneVisit.open(ev)
                seq.append(last)
                self.publish_event(seq.duplicate())
            else:
                # 기존에 있던 zone이 현 through event와 동일한 경우는 enter event를 무시한다.
                pass
            last.close(frame_index=ev.frame_index, ts=ev.ts)
        self.publish_event(seq.duplicate())
            
    def handle_track_deleted(self, ev:TrackDeleted):
        self.sequences.pop(ev.track_id, None)
        self.publish_event(ev)


class FinalZoneSequenceFilter(EventProcessor):
    def __init__(self) -> None:
        super().__init__()
        
        self.sequences:Dict[int,ZoneSequence] = dict()
    
    def close(self) -> None:
        self.sequences.clear()
        super().close()
        
    def handle_event(self, ev:Union[ZoneSequence,TrackDeleted]) -> None:
        if isinstance(ev, ZoneSequence):
            self.sequences[ev.track_id] = ev
        elif isinstance(ev, TrackDeleted):
            zseq = self.sequences.get(ev.track_id)
            if zseq:
                self.publish_event(zseq)