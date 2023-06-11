from __future__ import annotations
from collections.abc import Callable
from enum import Enum

from dna import ByteString
from dna.event import KafkaEvent, TrackEvent, TrackFeature, TrackletMotion, KafkaEventDeserializer, KafkaEventSerializer
from dna.support import iterables


class NodeEvent(Enum):
    TRACK_EVENT = ("track-events", TrackEvent)
    TRACK_FEATURE = ("track-features", TrackFeature)
    TRACK_MOTION = ("track-motions", TrackletMotion)

    def __init__(self, topic:str, event_type:type[KafkaEvent]) -> None:
        self.topic = topic
        self.event_type = event_type

    @property
    def serializer(self) -> KafkaEventSerializer:
        return self.event_type.serialize

    @property
    def deserializer(self) -> KafkaEventDeserializer:
        return self.event_type.deserialize

    @classmethod
    def topics(cls) -> list[str]:
        return [node_ev.topic for node_ev in cls]

    @classmethod
    def from_topic(cls, topic:str) -> NodeEvent:
        return iterables.first(node_ev for node_ev in cls if node_ev.topic == topic)

    @classmethod
    def from_event_type(cls, event_type:type[KafkaEvent]) -> NodeEvent:
        return iterables.first(node_ev for node_ev in cls if node_ev.event_type == event_type)