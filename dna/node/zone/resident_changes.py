from __future__ import annotations
from typing import Union, Set, Dict
from dataclasses import dataclass, field

from ..event_processor import EventProcessor
from .types import ZoneEvent, ResidentChanged, TrackDeleted

import logging
LOGGER = logging.getLogger('dna.node.zone.Residents')


@dataclass
class Residents:
    track_ids: Set[str]
    frame_index: int
    ts: int
    
    def add_resident(self, track_id:str) -> bool:
        if track_id not in self.track_ids:
            self.track_ids.add(track_id)
            return True
        else:
            return False
    
    def remove_resident(self, track_id:str) -> bool:
        if track_id in self.track_ids:
            self.track_ids.discard(track_id)
            return True
        else:
            return False
            

class ResidentChanges(EventProcessor):
    __slots__ = ( 'residents' )

    def __init__(self) -> None:
        EventProcessor.__init__(self)
        
        self.residents:Dict[str,Residents] = dict()

    def close(self) -> None:
        self.residents.clear()
        super().close()

    def handle_event(self, ev:Union[ZoneEvent,TrackDeleted]) -> None:
        if isinstance(ev, ZoneEvent):
            self.handle_zone_event(ev)
        elif isinstance(ev, TrackDeleted):
            self.handle_track_deleted(ev)
        else:
            self._publish_event(ev)
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.info(f'unknown event: {ev}')
                
    def handle_zone_event(self, ev:ZoneEvent) -> None:
        residents = self._get_residents(ev.zone_id)
        
        publish_event = False      
        if ev.is_entered() or ev.is_inside():
            publish_event = residents.add_resident(ev.track_id)
        elif ev.is_left() or ev.is_unassigned() or ev.is_through():
            publish_event = residents.remove_resident(ev.track_id)
        residents.frame_index =  ev.frame_index
        residents.ts = ev.ts
        if publish_event:
            self._publish_event(self._create_resident_changed(ev.zone_id, residents))
                
    def handle_track_deleted(self, deleted:TrackDeleted) -> None:
        # 제거된 track id를 갖는 residents를 검색하여 그 residents에서 삭제한다.
        track_id = deleted.track_id
        for zone_id, residents in self.residents.items():
            if residents.remove_resident(track_id):
                residents.frame_index =  deleted.frame_index
                residents.ts = deleted.ts
                self._publish_event(self._create_resident_changed(zone_id, residents))
                return
        self._publish_event(deleted)
        
    def _get_residents(self, zone_id:str) -> Residents:
        residents = self.residents.get(zone_id, None)
        if residents is None:
            residents = Residents(track_ids=set(), frame_index=0, ts=0)
            self.residents[zone_id] = residents
        return residents
    
    def _create_resident_changed(self, zone_id:str, residents:Residents) -> ResidentChanged:
        return ResidentChanged(zone_id=zone_id, residents=residents.track_ids,
                               frame_index=residents.frame_index, ts=residents.ts)