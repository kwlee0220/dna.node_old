from __future__ import annotations

from typing import Optional, List, NewType, Iterable, ByteString, Tuple, Any
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, asdict

import threading
import time
from datetime import timedelta
import json
import numpy as np

from dna import Point, Box
from dna.support import sql_utils
from dna.tracker import TrackState


NodeId = NewType('NodeId', str)
TrackId = NewType('TrackId', str)

_WGS84_PRECISION = 7
_DIST_PRECISION = 3


@dataclass(frozen=True, order=True, slots=True)
class TrackletId:
    node_id: NodeId
    track_id: TrackId
    
    def __iter__(self):
        return iter((self.node_id, self.track_id))
    
    def __repr__(self) -> str:
        return f'{self.node_id}[{self.track_id}]'


@dataclass(frozen=True)
class TimeElapsed:
    ts: int = field(default_factory=lambda: int(round(time.time() * 1000)))


class KafkaEvent(metaclass=ABCMeta):
    @abstractmethod
    def key(self) -> str: pass
    
    @abstractmethod
    def serialize(self) -> Any: pass
    

class KVKafkaEvent(KafkaEvent):
    def __init__(self, key:str, value:Any) -> None:
        self.key = key
        self.value = value
        
    def key(self) -> str:
        return self.key
    
    def serialize(self) -> Any:
        return self.value


def box_to_json(box:Box) -> List[float]:
    return [round(v, 2) for v in box.tlbr.tolist()] if box else None

def json_to_box(tlbr_list:Optional[Iterable[float]]) -> Box:
    return Box(tlbr_list) if tlbr_list else None


@dataclass(frozen=True, eq=True, order=False, repr=False, slots=True)
class TrackEvent(KafkaEvent):
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
    def from_row(row:Tuple) -> TrackEvent:
        return TrackEvent(node_id=row[1],
                            track_id=row[2],
                            state=TrackState.from_abbr(row[3]),
                            location=sql_utils.from_sql_box(row[4]),
                            world_coord=sql_utils.from_sql_point(row[5]),
                            distance=row[6],
                            zone_relation=row[7],
                            frame_index=row[8],
                            ts=row[9])
    
    def to_row(self) -> Tuple:
        return (self.node_id, self.track_id, self.state.abbr,
                sql_utils.to_sql_box(self.location.to_rint()),
                sql_utils.to_sql_point(self.world_coord),
                self.distance, self.zone_relation,
                self.frame_index, self.ts)

    @staticmethod
    def from_json(json_str:str) -> TrackEvent:
        json_obj = json.loads(json_str)

        world_coord = json_obj.get('world_coord', None)
        if world_coord is not None:
            world_coord = Point(world_coord)
        distance = json_obj.get('distance', None)
        zone_relation = json_obj.get('zone_relation', None)
        # detection_box = json_to_box(json_obj.get('detection_box', None))

        return TrackEvent(node_id=json_obj['node'],
                            track_id=json_obj['track_id'],
                            state=TrackState[json_obj['state']],
                            location=json_to_box(json_obj['location']),
                            world_coord=world_coord,
                            distance=distance,
                            zone_relation = zone_relation,
                            frame_index=json_obj['frame_index'],
                            ts=json_obj['ts'])

    def to_json(self) -> str:
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
    def deserialize(serialized:ByteString) -> TrackEvent:
        return TrackEvent.from_json(serialized.decode('utf-8'))

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
            
        return TrackEvent(node_id=node_id, track_id=track_id, state=state, location=loc,
                            frame_index=frame_idx, ts=ts, world_coord=world_coord, distance=dist)
    
    def __repr__(self) -> str:
        return (f"TrackEvent[id={self.node_id}[{self.track_id}]({self.state.abbr}), frame={self.frame_index}, loc={self.location}, ts={self.ts}]")

EOT:TrackEvent = TrackEvent(node_id=None, track_id=None, state=None, location=None,
                            world_coord=None, distance=None, frame_index=-1, ts=-1)

@dataclass(frozen=True, eq=True, slots=True)
class TrackDeleted:
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    ts: int = field(hash=False)

    def key(self) -> str:
        return self.node_id
    
    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: id={self.node_id}[{self.track_id}], ts={self.ts}")


from ..utils import utc2datetime
from .types import KafkaEvent
from .proto.reid_metrics_pb2 import TrackFeatureProto
class TrackFeature(KafkaEvent):
    __slots__ = ('node_id', 'track_id', '_bfeature', '_feature', 'zone_relation', 'ts')
    
    def __init__(self, **kwargs) -> None:
        self.node_id:str = kwargs['node_id']
        self.track_id:str = kwargs['track_id']
        self._bfeature:Optional[ByteString] = kwargs.get('bfeature')
        self._feature:Optional[np.ndarray] = kwargs.get('feature')
        self.zone_relation:str = kwargs.get('zone_relation')
        self.ts:int = kwargs['ts']
        
    def key(self) -> str:
        return self.node_id
    
    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)
    
    @property
    def feature(self) -> np.ndarray:
        if self._bfeature is None and self._feature is None:
            return None
        if self._feature is None:
            self._feature = np.frombuffer(self._bfeature, dtype=np.float32)
        return self._feature
        
    @property
    def bfeature(self) -> ByteString:
        if self._bfeature is None and self._feature is None:
            return None
        if not self._bfeature:
            self._bfeature = self._feature.tobytes()
        return self._bfeature
        
    @staticmethod
    def from_row(args) -> TrackFeature:
        return TrackFeature(node_id=args[0], track_id=args[1], bfeature=args[2], zone_relation=args[3], ts=args[4])
    
    def to_row(self) -> Tuple[str,str,ByteString,int]:
        return (self.node_id, self.track_id, self.bfeature, self.zone_relation, self.ts)
    
    def serialize(self) -> bytes:
        return self.to_bytes()
    
    def deserialize(binary_data:ByteString) -> TrackFeature:
        return TrackFeature.from_bytes(binary_data)
    
    def to_bytes(self) -> bytes:
        proto = TrackFeatureProto()
        proto.node_id = self.node_id
        proto.track_id = self.track_id
        if self.bfeature is not None:
            proto.bfeature = self.bfeature
        proto.zone_relation = self.zone_relation
        proto.ts = self.ts
        
        return proto.SerializeToString()
    
    @staticmethod
    def from_bytes(binary_data:bytes) -> TrackFeature:
        proto = TrackFeatureProto()
        proto.ParseFromString(binary_data)
        
        bfeature = proto.bfeature if proto.HasField('bfeature') else None
        return TrackFeature(node_id=proto.node_id, track_id=proto.track_id, bfeature=bfeature,
                            zone_relation=proto.zone_relation, ts=proto.ts)
        
    def __repr__(self) -> str:
        # dt = utc2datetime(self.ts)
        return f'{self.__class__.__name__}[id={self.node_id}[{self.track_id}], zone={self.zone_relation}, ts={self.ts}]'