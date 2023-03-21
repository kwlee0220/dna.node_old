from __future__ import annotations

from typing import Optional, List, NewType
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, asdict

import threading
import time
from datetime import timedelta
import json
import numpy as np

from dna import Point, Box
from dna.tracker import TrackState, ObjectTrack


NodeId = NewType('NodeId', str)
TrackId = NewType('TrackId', str)

_WGS84_PRECISION = 7
_DIST_PRECISION = 3


@dataclass(frozen=True)
class TimeElapsed:
    ts: int = field(default_factory=lambda: int(round(time.time() * 1000)))


class KafkaEvent(metaclass=ABCMeta):
    @abstractmethod
    def key(self) -> str: pass
    
    @abstractmethod
    def serialize(self) -> str: pass


@dataclass(frozen=True, eq=True, order=False, repr=False)    # slots=True
class TrackEvent(KafkaEvent):
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    state: TrackState   # tracking state
    location: Box = field(hash=False)
    frame_index: int
    ts: int = field(hash=False)
    world_coord: Optional[Point] = field(default=None, repr=False, hash=False)
    distance: Optional[float] = field(default=None, repr=False, hash=False)
    zone_relation: str = field(default=None)

    def key(self) -> str:
        return self.node_id.encode('utf-8')
    
    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def __lt__(self, other) -> bool:
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index == other.frame_index:
            return self.track_id < other.luid
        else:
            return False

    @staticmethod
    def from_track(node_id:NodeId, track:ObjectTrack) -> TrackEvent:
        return TrackEvent(node_id=node_id, track_id=track.id, state=track.state,
                        location=track.location, frame_index=track.frame_index, ts=int(track.timestamp * 1000))

    @staticmethod
    def from_json(json_str:str) -> TrackEvent:
        json_obj = json.loads(json_str)

        world_coord = json_obj.get('world_coord', None)
        if world_coord is not None:
            world_coord = Point.from_np(world_coord)
        distance = json_obj.get('distance', None)
        zone_rel = json_obj.get('zone_relation', None)

        return TrackEvent(node_id=json_obj['node'],
                            track_id=json_obj['track_id'],
                            state=TrackState[json_obj['state']],
                            location=Box.from_tlbr(np.array(json_obj['location'])),
                            frame_index=json_obj['frame_index'],
                            ts=json_obj['ts'],
                            world_coord=world_coord,
                            distance=distance,
                            zone_relation=zone_rel)

    def to_json(self) -> str:
        tlbr_expr = [round(v, 2) for v in self.location.tlbr.tolist()]
        serialized = {'node':self.node_id, 'track_id':self.track_id, 'state':self.state.name,
                    'location':tlbr_expr, 'frame_index':self.frame_index, 'ts': self.ts}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, _WGS84_PRECISION) for v in self.world_coord.to_tuple()]
        if self.distance is not None:
            serialized['distance'] = round(self.distance, _DIST_PRECISION)
        if self.zone_relation:
            serialized['zone_relation'] = self.zone_relation

        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> str:
        tlbr_expr = [round(v, 2) for v in self.location.tlbr.tolist()]
        serialized = {'node':self.node_id, 'track_id':self.track_id, 'state':self.state.name,
                    'location':tlbr_expr, 'frame_index':self.frame_index, 'ts': self.ts}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, _WGS84_PRECISION) for v in self.world_coord.to_tuple()]
        if self.distance is not None:
            serialized['distance'] = round(self.distance, _DIST_PRECISION)
        if self.zone_relation:
            serialized['zone_relation'] = self.zone_relation

        return self.to_json().encode('utf-8')

    def updated(self, **kwargs) -> TrackEvent:
        fields = asdict(self)
        for key, value in kwargs.items():
            fields[key] = value
        return TrackEvent(**fields)

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
    def from_csv(csv: str):
        parts = csv.split(',')

        node_id = parts[0]
        track_id = parts[1]
        state = TrackState[parts[2]]
        loc = Box.from_tlbr(np.array([float(s) for s in parts[3:7]]))
        frame_idx = int(parts[7])
        ts = int(parts[8])
        xy_str = parts[9:11]
        if len(xy_str[0]) > 0:
            world_coord = Point.from_np(np.array([float(s) for s in xy_str]))
            dist = float(parts[11])
        else:
            world_coord = None
            dist = None
            
        return TrackEvent(node_id=node_id, track_id=track_id, state=state, location=loc,
                            frame_index=frame_idx, ts=ts, world_coord=world_coord, distance=dist)
    
    def __repr__(self) -> str:
        return (f"TrackEvent[id={self.track_id}({self.state.abbr}), frame={self.frame_index}, loc={self.location}]")

EOT:TrackEvent = TrackEvent(node_id=None, track_id=None, state=None, location=None,
                            world_coord=None, distance=None, frame_index=-1, ts=-1)