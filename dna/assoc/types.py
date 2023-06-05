from __future__ import annotations
from typing import Tuple

from dataclasses import dataclass, field

from dna.event.types import TrackletId

       
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