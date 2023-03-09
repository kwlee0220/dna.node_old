from __future__ import annotations

from typing import Optional, List
import dataclasses
from dataclasses import dataclass, field, asdict
import json

import numpy as np

from dna import Box, Point, utils
from dna.tracker import ObjectTrack, TrackState
from .kafka_event import KafkaEvent


_WGS84_PRECISION = 7
_DIST_PRECISION = 3
@dataclass(frozen=True, eq=True, order=False, repr=False)    # slots=True
class TrackEvent(KafkaEvent):
    node_id: str        # node id
    track_id: int       # tracking object id
    state: TrackState   # tracking state
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
            return self.track_id < other.luid
        else:
            return False

    @staticmethod
    def from_track(node_id:str, track:ObjectTrack) -> TrackEvent:
        return TrackEvent(node_id=node_id, track_id=track.id, state=track.state,
                        location=track.location, frame_index=track.frame_index, ts=int(track.timestamp * 1000))

    @staticmethod
    def from_json(json_str:str) -> TrackEvent:
        json_obj = json.loads(json_str)

        world_coord = json_obj.get('world_coord', None)
        if world_coord is not None:
            world_coord = Point.from_np(world_coord)
        distance = json_obj.get('distance', None)

        return TrackEvent(node_id=json_obj['node'],
                            track_id=json_obj['track_id'],
                            state=TrackState[json_obj['state']],
                            location=Box.from_tlbr(np.array(json_obj['location'])),
                            frame_index=json_obj['frame_index'],
                            ts=json_obj['ts'],
                            world_coord=world_coord,
                            distance=distance)

    def to_json(self) -> str:
        tlbr_expr = [round(v, 2) for v in self.location.tlbr.tolist()]
        serialized = {'node':self.node_id, 'track_id':self.track_id, 'state':self.state.name,
                    'location':tlbr_expr, 'frame_index':self.frame_index, 'ts': self.ts}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, _WGS84_PRECISION) for v in self.world_coord.to_tuple()]
        if self.distance is not None:
            serialized['distance'] = round(self.distance, _DIST_PRECISION)

        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> str:
        tlbr_expr = [round(v, 2) for v in self.location.tlbr.tolist()]
        serialized = {'node':self.node_id, 'track_id':self.track_id, 'state':self.state.name,
                    'location':tlbr_expr, 'frame_index':self.frame_index, 'ts': self.ts}
        if self.world_coord is not None:
            serialized['world_coord'] = [round(v, _WGS84_PRECISION) for v in self.world_coord.to_tuple()]
        if self.distance is not None:
            serialized['distance'] = round(self.distance, _DIST_PRECISION)

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
        luid = int(parts[1])
        state = TrackState[parts[2]]
        loc = Box.from_tlbr(np.array([float(s) for s in parts[3:7]]))
        frame_idx = int(parts[7])
        ts=int(parts[8])
        xy_str = parts[9:11]
        if len(xy_str[0]) > 0:
            world_coord = Point.from_np(np.array([float(s) for s in xy_str]))
            dist = float(parts[11])
        else:
            world_coord = None
            dist = None
            
        return TrackEvent(node_id=node_id, track_id=luid, state=state, location=loc,
                            frame_index=frame_idx, ts=ts, world_coord=world_coord, distance=dist)
    
    def __repr__(self) -> str:
        return (f"TrackEvent[id={self.track_id}({self.state.abbr}), frame={self.frame_index}, loc={self.location}]")

EOT:TrackEvent = TrackEvent(node_id=None, track_id=None, state=None, location=None,
                            world_coord=None, distance=None, frame_index=-1, ts=-1)


from dna import Frame
from dna.tracker import TrackProcessor
from .event_processor import EventQueue

class TrackEventSource(TrackProcessor, EventQueue):
    def __init__(self, node_id:str) -> None:
        TrackProcessor.__init__(self)
        EventQueue.__init__(self)

        self.node_id = node_id

    def track_started(self, tracker) -> None: pass
    def track_stopped(self, tracker) -> None:
        self.close()

    def process_tracks(self, tracker, frame:Frame, tracks:List[ObjectTrack]) -> None:
        for track in tracks:
            self.publish_event(TrackEvent.from_track(self.node_id, track))