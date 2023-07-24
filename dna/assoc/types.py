from __future__ import annotations

from dataclasses import dataclass, field
import json

from dna import ByteString, Point, TrackletId, TrackletId
from dna.event import KafkaEvent


@dataclass(frozen=True, eq=False, order=False, repr=False)   # slots=True
class LocalTrack:
    node: str           # node id
    track_id: str       # track id
    location: Point = field(hash=False)
    ts: int = field(hash=False)

    def key(self) -> str:
        return self.node
    
    def is_same_track(self, ltrack:LocalTrack) -> bool:
        return self.node == ltrack.node and self.track_id == ltrack.track_id

    @staticmethod
    def from_json_object(json_obj:dict[str,object]) -> LocalTrack:
        return LocalTrack(node=json_obj['node'], track_id=json_obj['track_id'],
                            location=Point(json_obj.get('location')),
                            ts=json_obj['ts'])

    def to_json_object(self) -> str:
        return {
            'node': self.node,
            'track_id': self.track_id,
            'location': list(self.location.xy),
            'ts': self.ts
        }

    def __repr__(self) -> str:
        return f"{self.node}[{self.track_id}]:{self.location}#{self.ts}"


@dataclass(frozen=True, eq=False, order=False, repr=False)   # slots=True
class GlobalTrack(KafkaEvent):
    node: str                                       # node id
    track_id: str                                   # track id
    location: Point = field(hash=False)
    overlap_area: str = field(hash=False)           # overlap area id (nullable)
    support: list[LocalTrack] = field(hash=False)   # nullable
    ts: int = field(hash=False)

    def key(self) -> str:
        return self.node
    
    def is_same_track(self, ltrack:LocalTrack) -> bool:
        return self.node == ltrack.node and self.track_id == ltrack.track_id

    @staticmethod
    def from_json(json_str:str) -> GlobalTrack:
        json_obj = json.loads(json_str)
        
        support_json_obj = json_obj.get('support')
        support = None if support_json_obj is None \
                        else [LocalTrack.from_json_object(sj) for sj in support_json_obj]
        
        support = [LocalTrack.from_json_object(sj) for sj in json_obj['support']]
        return GlobalTrack(node=json_obj['node'], track_id=json_obj['track_id'],
                            location=Point(json_obj.get('location')),
                            overlap_area=json_obj.get('overlap_area'),
                            support=support,
                            ts=json_obj['ts'])

    def to_json(self) -> str:
        serialized = {
            'node': self.node,
            'track_id': self.track_id,
            'location': list(self.location.xy),
        }
        if self.overlap_area:
            serialized['overlap_area'] = self.overlap_area
            serialized['support'] = [s.to_json_object() for s in self.support]
        serialized['ts'] = self.ts
            
        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> bytes:
        return self.to_json().encode('utf-8')

    @staticmethod
    def deserialize(serialized:ByteString) -> GlobalTrack:
        return GlobalTrack.from_json(serialized.decode('utf-8'))

    def __repr__(self) -> str:
        support_str = ""
        if self.overlap_area:
            lt_str = '-'.join(f"{lt.node}[{lt.track_id}]" for lt in self.support)
            support_str = f" - {self.overlap_area}:{{{lt_str}}}"
        return f"{self.node}[{self.track_id}]:{self.location}#{self.ts}{support_str}"


@dataclass(frozen=True)
class TrajectoryFragment:
    id: int
    tracklet: TrackletId
        
    @staticmethod
    def from_row(args) -> TrajectoryFragment:
        tracklet = TrackletId(args[1], args[2])
        return TrajectoryFragment(id, tracklet)
    
    def to_row(self) -> tuple[str,str,str]:
        return (self.id, self.tracklet.node_id, self.tracklet.track_id)