from typing import Any, Optional

from omegaconf import OmegaConf

from dna.tracker import TrackProcessor
from .event_processor import EventQueue


# class EventPublisher:
#     __slots__ = 'pubsub', 'channel', 'nsubscribers'

#     def __init__(self, pubsub: PubSub, channel: str) -> None:
#         self.pubsub = pubsub
#         self.channel = channel
#         self.nsubscribers = 0

#     def publish(self, ev: Any) -> None:
#         if self.nsubscribers > 0:
#             self.pubsub.publish(self.channel, ev)

#     def subscribe(self) -> Queue:
#         self.nsubscribers += 1
#         return self.pubsub.subscribe(self.channel)

#     def __repr__(self) -> str:
#         return f'EventPublisher[channel={self.channel}]'


_DEFAULT_MIN_PATH_LENGTH=10

def load_publishing_pipeline(node_id: str, publishing_conf: OmegaConf) -> TrackProcessor:
    from .track_event import TrackEventSource
    from .kafka_event_publisher import KafkaEventPublisher
    from .event_processor import PrintTrackEvent

    source = TrackEventSource(node_id)
    queue = source
    
    # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
    from .refine_track_event import RefineTrackEvent
    refine = RefineTrackEvent()
    queue.add_listener(refine)
    queue = refine

    # drop too-short tracks of an object
    min_path_length = publishing_conf.get('min_path_length', _DEFAULT_MIN_PATH_LENGTH)
    if min_path_length > 0:
        from .drop_short_trail import DropShortTrail
        drop_short_path = DropShortTrail(min_path_length)
        queue.add_listener(drop_short_path)
        queue = drop_short_path

    # attach world-coordinates to each track
    if publishing_conf.get('attach_world_coordinates') is not None:
        from .world_transform import WorldTransform
        world_coords = WorldTransform(publishing_conf.attach_world_coordinates)
        queue.add_listener(world_coords)
        queue = world_coords

    if publishing_conf.get('kafka', None) is not None:
        queue.add_listener(KafkaEventPublisher(publishing_conf.kafka))
    if publishing_conf.get('output', None) is not None:
        queue.add_listener(PrintTrackEvent(publishing_conf.output))
        
    return source