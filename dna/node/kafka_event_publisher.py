
from kafka import KafkaProducer
from omegaconf import OmegaConf

from pubsub import Queue
from .track_event import TrackEvent
from .event_processor import EventProcessor
from .kafka_event import KafkaEvent


class KafkaEventPublisher(EventProcessor):
    def __init__(self, in_queue: Queue, topic:str, conf:OmegaConf) -> None:
        super().__init__(in_queue)

        self.producer = KafkaProducer(bootstrap_servers=conf.bootstrap_servers)
        self.topic = topic

    def close(self) -> None:
        self.producer.flush()

    def handle_event(self, ev: KafkaEvent) -> None:
        key = ev.key()
        value = ev.serialize()
        self.producer.send(self.topic, value=value, key=key)
        # print(value)

    def flush(self) -> None:
        self.producer.flush()