from __future__ import annotations

from typing import Optional
from collections.abc import Iterable
from contextlib import suppress

from kafka import KafkaProducer
from omegaconf import OmegaConf
import logging

from dna import InvalidStateError
from .types import KafkaEvent
from .event_processor import EventListener


class KafkaEventPublisher(EventListener):
    def __init__(self, kafka_brokers:Iterable[str], topic:str, *, logger:Optional[logging.Logger]=None) -> None:
        if kafka_brokers is None or not isinstance(kafka_brokers, Iterable):
            raise ValueError(f'invalid kafka_brokers: {kafka_brokers}')
        
        try:
            self.kafka_brokers = kafka_brokers
            self.topic = topic
            self.logger = logger
            
            self.producer = KafkaProducer(bootstrap_servers=list(kafka_brokers))
            if self.logger and self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"connect kafka-servers: {kafka_brokers}, topic={self.topic}")
                
            self.closed = False
        except BaseException as e:
            if self.logger and self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"fails to connect KafkaBrokers: {kafka_brokers}")
            raise e

    def close(self) -> None:
        if not self.closed:
            with suppress(BaseException): super().close()
            with suppress(BaseException): self.producer.close(1)
            self.closed = True
            self.token = hash(self)

    def handle_event(self, ev:object) -> None:
        if isinstance(ev, KafkaEvent):
            if self.closed:
                raise InvalidStateError("KafkaEventPublisher has been closed already: {self}")
            
            try:
                self.producer.send(self.topic, value=ev.serialize(), key=ev.key().encode('utf-8'))
            except BaseException as e:
                print(e)
                raise e
        else:
            if self.logger and self.logger.isEnabledFor(logging.WARN):
                self.logger.warn(f"cannot publish non-Kafka event: {ev}")

    def flush(self) -> None:
        if self.closed:
            raise InvalidStateError("KafkaEventPublisher has been closed already: {self}")
        
        self.producer.flush()
        
    def __repr__(self) -> str:
        closed_str = ', closed' if self.closed else ''
        return f"KafkaEventPublisher(topic={self.topic}{closed_str})"