from __future__ import annotations
from typing import List, Union, Tuple

import numbers
import logging
import numpy.typing as npt
from shapely.geometry.base import BaseGeometry
import shapely.geometry as geometry
from omegaconf.omegaconf import OmegaConf

from dna import Point, Box, Image, BGR, plot_utils
from dna.zone import Zone
from ..event_processor import EventProcessor
from .types import LineTrack, TrackDeleted, ZoneRelation, ZoneEvent


# class Zone:
#     def __init__(self, geom:BaseGeometry) -> None:
#         self.geom = geom

#     @staticmethod
#     def from_coords(coords:list) -> Zone:
#         if isinstance(coords[0], numbers.Number):
#             return Zone(geometry.box(coords).coords)
#             # return Zone(geometry.Polygon(Box.from_tlbr(coords).coords))
#         else:
#             npoints = len(coords)
#             if npoints > 2:
#                 return Zone(geometry.Polygon([tuple(c) for c in coords]))
#             elif npoints == 2:
#                 return Zone(geometry.LineString([tuple(c) for c in coords]))
    
#     @property
#     def coords(self):
#         return self.geom.coords

#     def covers(self, geom:BaseGeometry) -> bool:
#         return self.geom.covers(geom)

#     def covers_point(self, pt:Union[Point,Tuple,npt.ArrayLike]) -> bool:
#         xy = pt.to_tuple() if isinstance(pt, Point) else tuple(pt)
#         return self.geom.covers(Point(xy))
    
#     def intersects(self, geom:BaseGeometry):
#         return self.geom.intersects(geom)
    
#     def distance(self, geom:BaseGeometry) -> float:
#         return self.geom.distance(geom)

#     def draw(self, convas:Image, color:BGR, line_thickness=2) -> Image:
#         if isinstance(self.geom, geometry.LineString):
#             return plot_utils.draw_line_string(convas, self.geom.coords, color=color, line_thickness=line_thickness)
#         else:
#             return plot_utils.draw_polygon(convas, list(self.geom.exterior.coords), color=color, line_thickness=line_thickness)

#     def __repr__(self) -> str:
#         return repr(self.geom)

#     @staticmethod
#     def find_covering_zone(pt:Point, zones:List[Zone]):
#         for idx, zone in enumerate(zones):
#             if zone.covers_point(pt):
#                 return idx
#         return -1

class ZoneEventGenerator(EventProcessor):
    def __init__(self, named_zones:OmegaConf, logger:logging.Logger) -> None:
        EventProcessor.__init__(self)
        self.zones = {str(zid):Zone.from_coords(zone_expr, as_line_string=True) for zid, zone_expr in named_zones.items()}
        self.logger = logger

    def handle_event(self, ev:LineTrack) -> None:
        if isinstance(ev, LineTrack):
            self.handle_line_track(ev)
        else:
            self.publish_event(ev)
        
    def handle_line_track(self, line_track:LineTrack) -> None:
        zone_events:List[ZoneEvent] = []
        for zid, zone in self.zones.items(): 
            if zone.intersects(line_track.line):
                rel = self.get_relation(zone, line_track.line)
                zone_events.append(self.to_zone_line_corss(rel, zid, line_track))

        # 특정 zone과 교집합이 없는 경우는 UNASSIGNED 이벤트를 발송함
        if len(zone_events) == 0:
            self.publish_event(ZoneEvent.UNASSIGNED(line_track))
        elif len(zone_events) == 1:
            # 가장 흔한 케이스로 1개의 zone과 연관된 경우는 바로 해당 event를 발송
            if self.logger.isEnabledFor(logging.DEBUG):
                if rel == ZoneRelation.Entered or rel == ZoneRelation.Left:
                    self.logger.debug(f'{zone_events[0]}')
            self.publish_event(zone_events[0])
        else:
            # 한 line에 여러 zone event가 발생 가능하기 때문에 이 경우 zone event 발생 순서를 조정함.
            #

            # 일단 left event가 존재하는가 확인하여 이를 첫번째로 발송함.
            left_idxes = [idx for idx, zone_ev in enumerate(zone_events) if zone_ev.relation == ZoneRelation.Left]
            for idx in left_idxes:
                left_event = zone_events.pop(idx)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'{left_event}')
                self.publish_event(left_event)

            from dna.utils import split_list
            enter_events, through_events = split_list(zone_events, lambda ev: ev.relation == ZoneRelation.Entered)
            if len(through_events) == 1:
                self.publish_event(through_events[0])
            elif len(through_events) > 1:
                start_pt = geometry.Point(line_track.line.coords[0])
                # line의 시작점을 기준으로 through된 zone과의 거리를 구한 후, 짧은 순서로 정렬시켜 event를 발송함
                zone_dists = [(idx, self.zones[through_ev.zone_id].distance(start_pt)) for idx, through_ev in enumerate(through_events)]
                zone_dists.sort(key=lambda zd: zd[1])
                for idx, dist in zone_dists:
                    self.publish_event(through_events[idx])

            # 마지막으로 enter event가 존재하는가 확인하여 이들을 발송함.
            for enter_ev in enter_events:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'{enter_ev}')
                self.publish_event(enter_ev)
        
    def handle_track_deleted(self, ev:TrackDeleted) -> None:
        self.publish_event(ev)
        
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
        
    def to_zone_line_corss(self, rel:ZoneRelation, zone_id:str, track:LineTrack) -> ZoneEvent:
        return ZoneEvent(track_id=track.track_id, relation=rel, zone_id=zone_id,
                         frame_index=track.frame_index, ts=track.ts)