from __future__ import annotations
from typing import Tuple

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrackletId:
    node_id: str
    track_id: str
    
    def __repr__(self) -> str:
        return f'{self.node_id}[{self.track_id}]'

       
@dataclass(frozen=True)
class Association:
    tracklet1: TrackletId
    tracklet2: TrackletId
    distance: float = field(compare=False)
        
    @staticmethod
    def from_row(args) -> Association:
        tracklet1 = TrackletId(args[0], args[1])
        tracklet2 = TrackletId(args[2], args[3])
        return Association(tracklet1, tracklet2, args[4])
    
    def to_row(self) -> Tuple[str,str,str,str,float]:
        return (self.tracklet1.node_id, self.tracklet1.track_id,
                self.tracklet2.node_id, self.tracklet2.track_id,
                self.distance)
    
    def __repr__(self) -> str:
        return f'{self.tracklet1} <-> {self.tracklet2}: {self.distance:.03f}'

       
@dataclass(frozen=True)
class TrajectoryFragment:
    id: int
    tracklet: TrackletId
        
    @staticmethod
    def from_row(args) -> TrajectoryFragment:
        tracklet = TrackletId(args[1], args[2])
        return TrajectoryFragment(id, tracklet)
    
    def to_row(self) -> Tuple[str,str,str]:
        return (self.id, self.tracklet.node_id, self.tracklet.track_id)