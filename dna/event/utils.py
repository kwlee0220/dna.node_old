from __future__ import annotations

from typing import Union, Optional, TypeVar
from collections.abc import Iterator, Sequence, Callable, Generator

from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import NoBrokersAvailable

# from dna import NodeId
from dna.support import iterables
from dna.event import KafkaEventDeserializer, NodeTrack, TrackFeature
from .types import KafkaEvent, Timestamped

T = TypeVar("T")


def open_kafka_producer(brokers:list[str], *,
                        key_serializer:Callable[[str],bytes]=lambda k: k.encode('utf-8')) -> KafkaProducer:
    try:
        return KafkaProducer(bootstrap_servers=brokers, key_serializer=key_serializer)
    except NoBrokersAvailable as e:
        raise NoBrokersAvailable(f'fails to connect to Kafka: server={brokers}')


def read_topics(consumer:KafkaConsumer, **poll_args) -> Generator[ConsumerRecord, None, None]:
    org_timeout_ms = None
    if 'initial_timeout_ms' in poll_args:
        org_timeout_ms = poll_args['timeout_ms']
        poll_args['timeout_ms']  = poll_args['initial_timeout_ms']
        del poll_args['initial_timeout_ms']
        
    stop_on_poll_timeout = poll_args.pop('stop_on_poll_timeout', False)
        
    while True:
        partitions = consumer.poll(**poll_args)
        if partitions:
            for part_info, partition in partitions.items():
                for record in partition:
                    yield record
            if org_timeout_ms is not None:
                poll_args['timeout_ms'] = org_timeout_ms
        elif stop_on_poll_timeout:
            break


def open_kafka_consumer(brokers:list[str], offset_reset:str,
                        *,
                        key_deserializer:Callable[[bytes],str]=lambda k:k.decode('utf-8')) -> KafkaConsumer:
    try:
        return KafkaConsumer(bootstrap_servers=brokers, auto_offset_reset=offset_reset,
                                key_deserializer=key_deserializer)
    except NoBrokersAvailable as e:
        raise NoBrokersAvailable(f'fails to connect to Kafka: server={brokers}')

def process_kafka_events(events:Union[Iterator[Timestamped],Sequence[Timestamped]],
                         processor:Callable[[Timestamped],None],
                         *,
                         sync:bool=False,
                         show_progress:bool=False,
                         max_wait_ms:Optional[int]=None):
    import time
    from tqdm import tqdm
    
    now = round(time.time() * 1000)
    if isinstance(events, Sequence):
        delta = now - events[0].ts
    elif isinstance(events, Iterator):
        events = iterables.to_peekable(events)
        delta = now - events.peek().ts
    else:
        raise ValueError(f"invalid events")
    
    last_ts = -1
    if show_progress:
        events = tqdm(events)
    for ev in events:
        if sync:
            sleep_ms = (ev.ts - last_ts) if last_ts > 0 else 0
            if max_wait_ms:
                sleep_ms = min(sleep_ms, max_wait_ms)
            if sleep_ms > 20:
                time.sleep((sleep_ms)/1000)
            last_ts = ev.ts
        processor(ev)

def publish_kafka_events(producer:KafkaProducer,
                         events:Union[Iterator[KafkaEvent],Sequence[KafkaEvent]],
                         topic_finder:Optional[Callable[[KafkaEvent],str]]=None,
                         sync:bool=False,
                         show_progress:bool=False):
    import time
    from tqdm import tqdm
    
    now = round(time.time() * 1000)
    if isinstance(events, Sequence):
        delta = now - events[0].ts
    elif isinstance(events, Iterator):
        events = iterables.to_peekable(events)
        delta = now - events.peek().ts
    else:
        raise ValueError(f"invalid events")
    
    if show_progress:
        events = tqdm(events)
    for ev in events:
        if sync:
            now = round(time.time() * 1000)
            now_expected = ev.ts + delta
            sleep_ms = now_expected - now
            if sleep_ms > 20:
                time.sleep((sleep_ms)/1000)
                
        topic = topic_finder(ev)
        print(f'topic={topic}, value={ev}')
        # producer.send(topic, value=ev.serialize(), key=ev.key())

            
def read_pickle_event_file(file:str) -> Generator[KafkaEvent, None, None]:
    import pickle
    with open(file, 'rb') as fp:
        try:
            while True:
                yield pickle.load(fp)
        except EOFError:
            return

def read_json_event_file(file:str, event_type:type[T]) -> Generator[T, None, None]:
    import json
    with open(file) as f:
        for line in f.readlines():
            yield event_type.from_json(line)
        
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
        return read_pickle_event_file(file)
    else:
        raise ValueError(f"invalid event type: {event_type}")