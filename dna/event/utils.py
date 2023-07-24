from __future__ import annotations

from typing import Union, Optional, TypeVar
from collections.abc import Iterator, Sequence, Callable, Generator

import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import NoBrokersAvailable

from dna.support import iterables
from dna.event import KafkaEventDeserializer
from .types import KafkaEvent

T = TypeVar("T")


def open_kafka_producer(brokers:list[str]) -> KafkaProducer:
    try:
        return KafkaProducer(bootstrap_servers=brokers)
    except NoBrokersAvailable as e:
        raise NoBrokersAvailable(f'fails to connect to Kafka: server={brokers}')


def open_kafka_consumer(brokers:list[str],
                        offset:str,
                        *,
                        key_deserializer:Optional[Callable[[bytes],str]]=None) -> KafkaConsumer:
    try:
        return KafkaConsumer(bootstrap_servers=brokers, auto_offset_reset=offset,
                                key_deserializer=key_deserializer)
    except NoBrokersAvailable as e:
        raise NoBrokersAvailable(f'fails to connect to Kafka: server={brokers}')


def publish_kafka_events(producer:KafkaProducer, events:Union[Iterator[KafkaEvent],Sequence[KafkaEvent]],
                         topic_finder:Optional[Callable[[KafkaEvent],str]]=None, sync:bool=False):
    now = round(time.time() * 1000)
    if isinstance(events, Sequence):
        delta = now - events[0].ts
    elif isinstance(events, Iterator):
        events = iterables.to_peekable(events)
        delta = now - events.peek().ts
    else:
        raise ValueError(f"invalid events")
    
    for ev in events:
        if sync:
            now = round(time.time() * 1000)
            now_expected = ev.ts + delta
            sleep_ms = now_expected - now
            if sleep_ms > 20:
                time.sleep((sleep_ms)/1000)
                
        topic = topic_finder(ev)
        producer.send(topic, value=ev.serialize(), key=ev.key().encode('utf-8'))


def read_topics(consumer:KafkaConsumer, **poll_args) -> Generator[ConsumerRecord, None, None]:
    mapping:dict[str,int] = dict()
    while True:
        partitions = consumer.poll(**poll_args)
        if partitions:
            for part_info, partition in partitions.items():
                print(f"partition={part_info.partition}, count={len(partition)}")
                # for record in partition:
                #     yield record
        else:
            break


def read_json_event_file(file:str, event_type:type[T]) -> Generator[T, None, None]:
    import json
    with open(file) as f:
        for line in f.readlines():
            yield event_type.from_json(line)
            
def read_pickle_event_file(file:str) -> Generator[KafkaEvent, None, None]:
    import pickle
    with open(file, 'rb') as fp:
        try:
            while True:
                yield pickle.load(fp)
        except EOFError:
            return
        
def read_event_file(file:str,
                    *,
                    event_type:Optional[type[KafkaEvent]]=None) -> Generator[KafkaEvent, None, None]:
    from pathlib import Path
    suffix = Path(file).suffix[1:]
    if suffix == 'json':
        if not event_type:
            raise ValueError(f"event type is not specified")
        return read_json_event_file(file, event_type)
    elif suffix == 'pickle':
        events_gen = read_pickle_event_file(file)
        if event_type:
            events_gen = filter(lambda ev: isinstance(ev, event_type), events_gen)
        return events_gen
    else:
        raise ValueError(f"invalid event type: {event_type}")