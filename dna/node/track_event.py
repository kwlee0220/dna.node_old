from __future__ import annotations

from typing import Optional, ClassVar
from dataclasses import dataclass, field, asdict
import json

import numpy as np

from dna import Box, Point
from dna.track import Track
from dna.track.tracker import TrackState
from .kafka_event import KafkaEvent


@dataclass(frozen=True, eq=True, order=False, repr=False)    # slots=True
class TrackEvent(KafkaEvent):
    node_id: str
    luid: int
    state: TrackState
    location: Box = field(hash=False)
    frame_index: int
    ts: int = field(hash=False)
    world_coord: Optional[Point] = field(default=None, repr=False, hash=False)
    distance: Optional[float] = field(default=None, repr=False, hash=False)

    def key(self) -> str:
        return self.node_id.encode('utf-8')

    def __lt__(self, other) -> bool:
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index == other.frame_index:
            return self.luid < other.luid
        else:
            return False

    @staticmethod
    def from_track(node_id:str, track:Track) -> TrackEvent:
        return TrackEvent(node_id=node_id, luid=track.id, state=track.state, location=track.location,
                        frame_index=track.frame_index, ts=int(track.ts * 1000))

    @staticmethod
    def from_json(json_str:str) -> TrackEvent:
        json_obj = json.loads(json_str)

        world_coord = json_obj.get('world_coord', None)
        if world_coord is not None:
            world_coord = Point.from_np(world_coord)
        distance = json_obj.get('distance', None)

        return TrackEvent(node_id=json_obj['node'],
                            luid=json_obj['luid'],
                            state=TrackState[json_obj['state']],
                            location=Box.from_tlbr(np.array(json_obj['location'])),
                            frame_index=json_obj['frame_index'],
                            ts=json_obj['ts'],
                            world_coord=world_coord,
                            distance=distance)

        print(type(json_obj))

    def to_json(self) -> str:
        tlbr_expr = [round(v, 2) for v in self.location.to_tlbr().tolist()]
        serialized = {'node':self.node_id, 'luid':self.luid, 'state':self.state.name,
                    'location':tlbr_expr, 'frame_index':self.frame_index, 'ts': self.ts}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, 3) for v in self.world_coord.to_tuple()]
        if self.distance is not None:
            serialized['distance'] = round(self.distance,2)

        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> str:
        tlbr_expr = [round(v, 2) for v in self.location.to_tlbr().tolist()]
        serialized = {'node':self.node_id, 'luid':self.luid, 'state':self.state.name,
                    'location':tlbr_expr, 'frame_index':self.frame_index, 'ts': self.ts}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, 3) for v in self.world_coord.to_tuple()]
        if self.distance is not None:
            serialized['distance'] = round(self.distance,2)

        return self.to_json().encode('utf-8')

    def updated(self, **kwargs) -> TrackEvent:
        fields = asdict(self)
        for key, value in kwargs.items():
            fields[key] = value
        return TrackEvent(**fields)

    def to_csv(self) -> str:
        vlist = [self.node_id, self.luid, self.state.name] \
                + self.location.to_tlbr().tolist() \
                + [self.frame_index, self.ts]
        if self.world_coord is not None:
            vlist += np.round(self.world_coord.xy, 3).tolist() + [round(self.distance, 3)]
        else:
            vlist += ['', '']

        return ','.join([str(v) for v in vlist])

    @staticmethod
    def from_csv(csv: str):
        parts = csv.split(',')

        node_id = parts[0]
        luid = int(parts[1])
        state = TrackState[parts[2]]
        loc = Box.from_tlbr(np.array([float(s) for s in parts[3:7]]))
        frame_idx = int(parts[7])
        ts=int(parts[8])
        xy_str = parts[9:11]
        if len(xy_str[0]) > 0:
            wcoord = Point.from_np(np.array([float(s) for s in xy_str]))
            dist = float(parts[11])
        else:
            wcoord = None
            dist = None
            
        return TrackEvent(node_id=node_id, luid=luid, state=state, location=loc,
                            frame_index=frame_idx, ts=ts, world_coord=wcoord, distance=dist)
    
    def __repr__(self) -> str:
        return (f"TrackEvent[node={self.node_id}, id={self.luid}, loc={self.location}, "
                f"frame={self.frame_index}, ts={self.ts}]")

EOT:TrackEvent = TrackEvent(node_id=None, luid=None, state=None, location=None,
                            world_coord=None, distance=None, frame_index=-1, ts=-1)