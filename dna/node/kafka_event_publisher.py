from __future__ import annotations
from typing import Optional

from kafka import KafkaProducer
from omegaconf import OmegaConf
import logging

from .types import KafkaEvent
from .event_processor import EventListener


class KafkaEventPublisher(EventListener):
    def __init__(self, bootstrap_servers:str, topic:str, *, logger:Optional[logging.Logger]=None) -> None:
        try:
            self.topic = topic
            self.logger = logger
            
            self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"connect kafka-servers: {bootstrap_servers}, topic={self.topic}")
        except BaseException as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"fails to connect KafkaBrokers: {bootstrap_servers}")
            raise e

    def close(self) -> None:
        super().close()
        self.producer.close(1)
        
    @classmethod
    def from_conf(cls, conf:OmegaConf, *, logger:Optional[logging.Logger]=None) -> KafkaEventPublisher:
        bootstrap_servers = conf.bootstrap_servers
        topic = conf.topic
        return cls(bootstrap_servers, topic, logger=logger)

    def handle_event(self, ev:KafkaEvent) -> None:
        if isinstance(ev, KafkaEvent):
            key = ev.key()
            value = ev.serialize()
            self.producer.send(self.topic, value=value, key=key.encode('utf-8'))

    def flush(self) -> None:
        self.producer.flush()