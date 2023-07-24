from __future__ import annotations

from typing import TypeAlias
from collections.abc import Callable
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import time

from dna import ByteString, NodeId, TrackId, TrackletId
from dna.track import TrackState


class KafkaEvent(metaclass=ABCMeta):
    @abstractmethod
    def key(self) -> str: pass
    
    @abstractmethod
    def serialize(self) -> bytes: pass
    
class SimpleKafkaEvent(KafkaEvent):
    def __init__(self, key:str, value:str) -> None:
        super().__init__()
        self.key = key
        self.value = value
        
    def key(self) -> str:
        return self.key
    
    def value(self) -> object:
        return self.value

KafkaEventDeserializer:TypeAlias = Callable[[ByteString], KafkaEvent]
KafkaEventSerializer:TypeAlias = Callable[[KafkaEvent], ByteString]


@dataclass(frozen=True, eq=True)    # slots=True
class TrackDeleted:
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    frame_index: int = field(hash=False)
    ts: int = field(hash=False)
    source:object = field(default=None)

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