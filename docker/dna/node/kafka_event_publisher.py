
from kafka import KafkaProducer
from omegaconf import OmegaConf
import logging

from .track_event import TrackEvent
from .event_processor import EventProcessor
from .kafka_event import KafkaEvent

LOGGER = logging.getLogger("dna.node.event")

class KafkaEventPublisher(EventProcessor):
    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)

        self.producer = KafkaProducer(bootstrap_servers=conf.bootstrap_servers)
        self.topic = conf.topic
        LOGGER.info(f"connect kafka-servers: {conf.bootstrap_servers}")

    def close(self) -> None:
        super().close()
        self.producer.close(1)

    def handle_event(self, ev: KafkaEvent) -> None:
        key = ev.key()
        value = ev.serialize()
        self.producer.send(self.topic, value=value, key=key)
        # print(value)

    def flush(self) -> None:
        self.producer.flush()