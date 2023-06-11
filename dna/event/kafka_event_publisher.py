from __future__ import annotations
from typing import Optional

from kafka import KafkaProducer
from omegaconf import OmegaConf
import logging

from dna.event import KafkaEvent
from .event_processor import EventListener


class KafkaEventPublisher(EventListener):
    def __init__(self, kafka_brokers:list[str], topic:str, *, logger:Optional[logging.Logger]=None) -> None:
        self.logger = logger
        
        if kafka_brokers is None or isinstance(kafka_brokers, list):
            raise ValueError(f'invalid kafka_brokers: {kafka_brokers}')
        
        try:
            self.topic = topic
            
            self.producer = KafkaProducer(bootstrap_servers=kafka_brokers)
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"connect kafka-servers: {kafka_brokers}, topic={self.topic}")
        except BaseException as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"fails to connect KafkaBrokers: {kafka_brokers}")
            raise e

    def close(self) -> None:
        super().close()
        self.producer.close(1)
        
    @classmethod
    def from_conf(cls, conf:OmegaConf, *, logger:Optional[logging.Logger]=None) -> KafkaEventPublisher:
        return cls(kafka_brokers=conf.kafka_brokers, topic=conf.topic, logger=logger)

    def handle_event(self, ev:KafkaEvent) -> None:
        if isinstance(ev, KafkaEvent):
            key = ev.key()
            value = ev.serialize()
            self.producer.send(self.topic, value=value, key=key.encode('utf-8'))

    def flush(self) -> None:
        self.producer.flush()