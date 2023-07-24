from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import json

import numpy as np

from dna import Box, Point, ByteString, NodeId, TrackId, TrackletId
from dna.track import TrackState
from dna.support import sql_utils
from .types import KafkaEvent


_WGS84_PRECISION = 7
_DIST_PRECISION = 3


@dataclass(frozen=True, eq=True, order=False, repr=False)   # slots=True
class NodeTrack(KafkaEvent):
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    state: TrackState   # tracking state
    location: Box = field(hash=False)
    frame_index: int
    ts: int = field(hash=False)
    world_coord: Optional[Point] = field(default=None, repr=False, hash=False)
    distance: Optional[float] = field(default=None, repr=False, hash=False)
    zone_relation: Optional[str] = field(default=None)
    detection_box: Optional[Box] = field(default=None)  # local-only

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def __lt__(self, other) -> bool:
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index == other.frame_index:
            return self.track_id < other.luid
        else:
            return False

    @staticmethod
    def from_row(row:tuple[str,str,TrackState,Box,Point,float,str,int,int]) -> NodeTrack:
        return NodeTrack(node_id=row[1],
                            track_id=row[2],
                            state=TrackState.from_abbr(row[3]),
                            location=sql_utils.from_sql_box(row[4]),
                            world_coord=sql_utils.from_sql_point(row[5]),
                            distance=row[6],
                            zone_relation=row[7],
                            frame_index=row[8],
                            ts=row[9])

    def to_row(self) -> tuple[str,str,str,str,str,float,str,int,int]:
        return (self.node_id, self.track_id, self.state.abbr,
                sql_utils.to_sql_box(self.location.to_rint()),
                sql_utils.to_sql_point(self.world_coord),
                self.distance, self.zone_relation,
                self.frame_index, self.ts)

    @staticmethod
    def from_json(json_str:str) -> NodeTrack:
        def json_to_box(tlbr_list:Optional[Iterable[float]]) -> Box:
            return Box(tlbr_list) if tlbr_list else None

        json_obj = json.loads(json_str)

        world_coord = json_obj.get('world_coord', None)
        if world_coord is not None:
            world_coord = Point(world_coord)
        distance = json_obj.get('distance', None)
        zone_relation = json_obj.get('zone_relation', None)
        # detection_box = json_to_box(json_obj.get('detection_box', None))

        return NodeTrack(node_id=json_obj['node'],
                            track_id=json_obj['track_id'],
                            state=TrackState[json_obj['state']],
                            location=json_to_box(json_obj['location']),
                            world_coord=world_coord,
                            distance=distance,
                            zone_relation = zone_relation,
                            frame_index=json_obj['frame_index'],
                            ts=json_obj['ts'])

    def to_json(self) -> str:
        def box_to_json(box:Box) -> list[float]:
            return [round(v, 2) for v in box.tlbr.tolist()] if box else None

        serialized = {'node':self.node_id, 'track_id':self.track_id, 'state':self.state.name,
                    'location':box_to_json(self.location)}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, _WGS84_PRECISION) for v in tuple(self.world_coord.xy)]
        if self.distance is not None:
            serialized['distance'] = round(self.distance, _DIST_PRECISION)
        if self.zone_relation:
            serialized['zone_relation'] = self.zone_relation
        serialized['frame_index'] = self.frame_index
        serialized['ts'] = self.ts

        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> str:
        return self.to_json().encode('utf-8')

    @staticmethod
    def deserialize(serialized:ByteString) -> NodeTrack:
        return NodeTrack.from_json(serialized.decode('utf-8'))

    def updated(self, **kwargs:object) -> NodeTrack:
        fields = asdict(self)
        for key, value in kwargs.items():
            fields[key] = value
        return NodeTrack(**fields)

    def to_csv(self) -> str:
        vlist = [self.node_id, self.track_id, self.state.name] \
                + self.location.tlbr.tolist() \
                + [self.frame_index, self.ts]
        if self.world_coord is not None:
            vlist += np.round(self.world_coord.xy, _WGS84_PRECISION).tolist() + [round(self.distance, _DIST_PRECISION)]
        else:
            vlist += ['', '']

        return ','.join([str(v) for v in vlist])

    @staticmethod
    def from_csv(csv: str) -> NodeTrack:
        parts = csv.split(',')

        node_id = parts[0]
        track_id = parts[1]
        state = TrackState[parts[2]]
        loc = Box([float(s) for s in parts[3:7]])
        frame_idx = int(parts[7])
        ts = int(parts[8])
        xy_str = parts[9:11]
        if len(xy_str[0]) > 0:
            world_coord = Point(np.array([float(s) for s in xy_str]))
            dist = float(parts[11])
        else:
            world_coord = None
            dist = None

        return NodeTrack(node_id=node_id, track_id=track_id, state=state, location=loc,
                            frame_index=frame_idx, ts=ts, world_coord=world_coord, distance=dist)

    def __repr__(self) -> str:
        return (f"TrackEvent[id={self.node_id}[{self.track_id}]({self.state.abbr}), "
                f"frame={self.frame_index}, loc={self.location}, ts={self.ts}]")