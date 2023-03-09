from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field

import logging
import shapely.geometry as geometry
from omegaconf.omegaconf import OmegaConf

from ..event_processor import EventQueue, EventListener
from .types import TrackDeleted, ZoneRelation, ZoneEvent, TimeElapsed, LocationChanged


@dataclass(frozen=True)
class ZoneLocations:
    zone_ids: Set[str]
    frame_index: int
    ts: float

    def __repr__(self) -> str:
        return f'zones={self.zone_ids}, frame={self.frame_index}'


class ZoneEventRefiner(EventListener):
    __slots__ = ('locations', 'zone_event_queue', 'location_event_queue', 'logger')

    def __init__(self, logger:logging.Logger) -> None:
        EventListener.__init__(self)

        self.locations:Dict[int,ZoneLocations]=dict()
        self.zone_event_queue = EventQueue()
        self.location_event_queue = EventQueue()
        self.logger = logger

    def close(self) -> None:
        locations = [(luid, zone_locs) for luid, zone_locs in self.locations.items()]
        for luid, zone_locs in locations:
            delete_ev = TrackDeleted(track_id=luid, frame_index=zone_locs.frame_index, ts=zone_locs.ts)
            self.handle_event(delete_ev)
                
    def handle_event(self, ev:Union[ZoneEvent,TrackDeleted]) -> None:
        if isinstance(ev, ZoneEvent):
            self.handle_zone_event(ev)
        elif isinstance(ev, TrackDeleted):
            self.handle_track_deleted(ev)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'unknown event: {ev}')
        
    def handle_zone_event(self, zone_ev:ZoneEvent) -> None:
        zone_ids = zlocs.zone_ids if (zlocs := self.locations.get(zone_ev.track_id)) else set()
        if zone_ev.relation == ZoneRelation.Unassigned:
            if zone_ids:
                # 이전에 소속되었던 모든 zone에 대해 Left event를 발생시킨다.
                for zid in zone_ids:
                    self.leave_zone(zone_ev, zone_id=zid)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'generate a LEFT(track={zone_ev.track_id}, zone={zone_ev.zone_id}, frame={zone_ev.frame_index})')
                    self.publish_left(zone_ev, zone_id=zid)
            self.zone_event_queue.publish_event(zone_ev)
        elif zone_ev.relation == ZoneRelation.Left:
            # 추적 물체가 해당 zone에 포함되지 않은 상태면, 먼저 해당 물체를 zone 안에 넣는 event를 추가한다.
            if zone_ev.zone_id not in zone_ids:
                self.enter_zone(zone_ev)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'generate a ENTERED(track={zone_ev.track_id}, zone={zone_ev.zone_id}, frame={zone_ev.frame_index})')
                self.publish_entered(zone_ev)
            self.leave_zone(zone_ev)
            self.zone_event_queue.publish_event(zone_ev)
        elif zone_ev.relation == ZoneRelation.Entered:
            if zone_ev.zone_id not in zone_ids:
                self.enter_zone(zone_ev)
                self.zone_event_queue.publish_event(zone_ev)
            else:
                if self.logger.isEnabledFor(logging.WARN):
                    self.logger.warn(f'ignore a ENTERED(track={zone_ev.track_id}, zone={zone_ev.zone_id}, frame={zone_ev.frame_index})')
        elif zone_ev.relation == ZoneRelation.Inside:
            if zone_ev.zone_id not in zone_ids:
                self.enter_zone(zone_ev)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'generate a ENTERED(track={zone_ev.track_id}, zone={zone_ev.zone_id}, frame={zone_ev.frame_index})')
                self.publish_entered(zone_ev)
            self.zone_event_queue.publish_event(zone_ev)
        elif zone_ev.relation == ZoneRelation.Through:
            if zone_ev.zone_id in zone_ids:
                self.leave_zone(zone_ev)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'replace the THROUGH(track={zone_ev.track_id}, zone={zone_ev.zone_id}) with '
                                      f'a LEFT(track={zone_ev.track_id}, zone={zone_ev.zone_id}, frame={zone_ev.frame_index})')
                self.publish_left(zone_ev)
            else:
                self.zone_event_queue.publish_event(zone_ev) 
        else:
            raise ValueError(f'invalid ZoneEvent: {zone_ev}')
                
    def handle_track_deleted(self, ev:TrackDeleted) -> None:
        track_id = ev.track_id
        
        # Zone에 위치한 상태에서 추적 물체가 delete된 경우에는 가짜로 LEFT event를 추가한다.
        zone_ids = zlocs.zone_ids if (zlocs := self.locations.get(track_id)) else set()
        if zone_ids:
            for zid in zone_ids.copy():
                self.leave_zone(ev, zone_id=zid)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'generate a LEFT(track={track_id}, zone={zid}, frame={ev.frame_index})')
                self.publish_left(ev, zone_id=zid)
        
        # 삭제된 track의 location 정보를 삭제한다
        self.locations.pop(track_id, None)
        # TrackDeleted 이벤트를 re-publish한다
        self.zone_event_queue.publish_event(ev)

    def enter_zone(self, zone_ev:ZoneEvent) -> ZoneLocations:
        zone_ids = zlocs.zone_ids if (zlocs := self.locations.get(zone_ev.zone_id)) else set()
        zone_ids.add(zone_ev.zone_id)
        zlocs = ZoneLocations(zone_ids, frame_index=zone_ev.frame_index, ts=zone_ev.ts)
        self.update_zone_locations(track_id=zone_ev.track_id, zlocs=zlocs)

    def leave_zone(self, zone_ev:ZoneEvent, zone_id:Optional[str]=None) -> ZoneLocations:
        zone_ids = zlocs.zone_ids if (zlocs := self.locations.get(zone_ev.track_id)) else set()
        zone_id = zone_id if zone_id else zone_ev.zone_id
        zone_ids.discard(zone_id)
        zlocs = ZoneLocations(zone_ids, frame_index=zone_ev.frame_index, ts=zone_ev.ts)
        self.update_zone_locations(track_id=zone_ev.track_id, zlocs=zlocs)
        
    def update_zone_locations(self, track_id:int, zlocs:ZoneLocations) -> None:
        self.locations[track_id] = zlocs
        location_changed = LocationChanged(track_id=track_id, zone_ids=zlocs.zone_ids, frame_index=zlocs.frame_index, ts=zlocs.ts)
        self.location_event_queue.publish_event(location_changed)
        
    def publish_entered(self, rel_ev:ZoneEvent, zone_id:Optional[str]=None) -> None:
        ev = ZoneEvent(track_id=rel_ev.track_id, relation=ZoneRelation.Entered,
                                     zone_id=zone_id if zone_id else rel_ev.zone_id,
                                     frame_index=rel_ev.frame_index, ts=rel_ev.ts)
        self.zone_event_queue.publish_event(ev)
        
    def publish_left(self, zone_ev:ZoneEvent, zone_id:Optional[str]=None) -> None:
        ev = ZoneEvent(track_id=zone_ev.track_id, relation=ZoneRelation.Left,
                                     zone_id=zone_id if zone_id else zone_ev.zone_id,
                                     frame_index=zone_ev.frame_index, ts=zone_ev.ts)
        self.zone_event_queue.publish_event(ev)
    