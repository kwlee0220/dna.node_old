from __future__ import annotations
from typing import Union, Optional

import numbers
import logging
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry
import shapely.geometry as geometry
from omegaconf.omegaconf import OmegaConf

from dna import Point, Box, Image, BGR
from dna.support import plot_utils
from dna.zone import Zone
from dna.event import EventProcessor, TrackDeleted
from .types import LineTrack, ZoneRelation, ZoneEvent


class ZoneEventGenerator(EventProcessor):
    def __init__(self, named_zones:OmegaConf, *, logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)
        self.zones = {str(zid):Zone.from_coords(zone_expr, as_line_string=True) for zid, zone_expr in named_zones.items()}
        self.logger = logger

    def handle_event(self, ev:object) -> None:
        if isinstance(ev, LineTrack):
            self.handle_line_track(ev)
        else:
            self._publish_event(ev)
        
    def handle_line_track(self, line_track:LineTrack) -> None:
        zone_events:list[ZoneEvent] = []
        # track의 첫번째 event인 경우는 point를 사용하고, 그렇지 않은 경우는 line을 사용하여 분석함.
        if line_track.is_point_track(): # point인 경우
            pt = line_track.end_point
            for zid, zone in self.zones.items():
                if zone.covers_point(pt):
                    zone_events.append(self.to_zone_event(ZoneRelation.Entered, zid, line_track))
                    break
        else:   # line인 경우
            for zid, zone in self.zones.items():
                if zone.intersects(line_track.line):
                    rel = self.get_relation(zone, line_track.line)
                    zone_events.append(self.to_zone_event(rel, zid, line_track))

        # 특정 zone과 교집합이 없는 경우는 UNASSIGNED 이벤트를 발송함
        if len(zone_events) == 0:
            self._publish_event(ZoneEvent.UNASSIGNED(line_track))
        elif len(zone_events) == 1:
            # 가장 흔한 케이스로 1개의 zone과 연관된 경우는 바로 해당 event를 발송
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                if rel == ZoneRelation.Entered or rel == ZoneRelation.Left:
                    self.logger.debug(f'{zone_events[0]}')
            self._publish_event(zone_events[0])
        else:
            # 한 line에 여러 zone event가 발생 가능하기 때문에 이 경우 zone event 발생 순서를 조정함.
            #

            # 일단 left event가 존재하는가 확인하여 이를 첫번째로 발송함.
            left_idxes = [idx for idx, zone_ev in enumerate(zone_events) if zone_ev.relation == ZoneRelation.Left]
            for idx in left_idxes:
                left_event = zone_events.pop(idx)
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'{left_event}')
                self._publish_event(left_event)

            from dna.utils import split_list
            enter_events, through_events = split_list(zone_events, lambda ev: ev.relation == ZoneRelation.Entered)
            if len(through_events) == 1:
                self._publish_event(through_events[0])
            elif len(through_events) > 1:
                def distance_to_cross(line, zone_id) -> geometry.Point:
                    overlap = self.zones[zone_id].intersection(line_track.line)
                    return overlap.distance(start_pt)

                start_pt = geometry.Point(line_track.line.coords[0])
                # line의 시작점을 기준으로 through된 zone과의 거리를 구한 후, 짧은 순서로 정렬시켜 event를 발송함
                zone_dists = [(idx, distance_to_cross(line_track.line, thru_ev.zone_id)) for idx, thru_ev in enumerate(through_events)]
                zone_dists.sort(key=lambda zd: zd[1])
                for idx, dist in zone_dists:
                    self._publish_event(through_events[idx])

            # 마지막으로 enter event가 존재하는가 확인하여 이들을 발송함.
            for enter_ev in enter_events:
                if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'{enter_ev}')
                self._publish_event(enter_ev)
        
    def handle_track_deleted(self, ev:TrackDeleted) -> None:
        self._publish_event(ev)
        
    def get_relation(self, zone:Zone, line:geometry.LineString) -> ZoneRelation:
        start_cond = zone.covers_point(line.coords[0])
        end_cond = zone.covers_point(line.coords[-1])
        if start_cond and end_cond:
            return ZoneRelation.Inside
        elif not start_cond and end_cond:
            return ZoneRelation.Entered
        elif start_cond and not end_cond:
            return ZoneRelation.Left
        else:
            return ZoneRelation.Through
        
    def to_zone_event(self, rel:ZoneRelation, zone_id:str, line:LineTrack) -> ZoneEvent:
        return ZoneEvent(track_id=line.track_id, relation=rel, zone_id=zone_id,
                         frame_index=line.frame_index, ts=line.ts, source=line.source)
        
    def __repr__(self) -> str:
        return f"GenerateZoneEvents[nzones={len(self.zones)}]"