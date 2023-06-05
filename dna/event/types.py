from __future__ import annotations
from typing import Any, NewType
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import time

from dna.track import TrackState


NodeId = NewType('NodeId', str)
TrackId = NewType('TrackId', str)


@dataclass(frozen=True, order=True) # slots=True
class TrackletId:
    node_id: NodeId
    track_id: TrackId

    def __iter__(self):
        return iter((self.node_id, self.track_id))

    def __repr__(self) -> str:
        return f'{self.node_id}[{self.track_id}]'
    

class KafkaEvent(metaclass=ABCMeta):
    @abstractmethod
    def key(self) -> str: pass
    
    @abstractmethod
    def serialize(self) -> Any: pass


@dataclass(frozen=True, eq=True)    # slots=True
class TrackDeleted:
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    frame_index: int = field(hash=False)
    ts: int = field(hash=False)
    source:Any = field(default=None)

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: id={self.node_id}[{self.track_id}], frame={self.frame_index}, ts={self.ts}")


@dataclass(frozen=True)
class TimeElapsed:
    ts: int = field(default_factory=lambda: int(round(time.time() * 1000)))