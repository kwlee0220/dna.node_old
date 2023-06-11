
from typing import Union
from collections.abc import Iterator, Callable

import dataclasses
import pickle
from pathlib import Path

from tqdm import tqdm
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from argparse import Namespace

from dna import initialize_logger
from dna.event import KafkaEvent, read_pickle_event_file, publish_kafka_events
from dna.node import NodeEvent
from scripts import *


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("file", help="events file (pickle format)")
    parser.add_argument("--kafka_brokers", default=['localhost:9092'], help="kafka broker hosts")
    parser.add_argument("--sync", action='store_true', default=False)
    parser.add_argument('--offsets', nargs='+', default=["0", "0", "0"], type=str, help='offsets')
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    offsets = [int(offset) for offset in args.offsets]
    def shift(ev:KafkaEvent) -> KafkaEvent:
        return dataclasses.replace(ev,
                                track_id=str(int(ev.track_id) + offsets[0]),
                                frame_index=(ev.frame_index) + offsets[1],
                                ts=(ev.ts) + offsets[2])
    
    print(f"loading events from the file '{args.file}'.")
    events = read_pickle_event_file(args.file)
    
    producer = None
    try:
        producer = KafkaProducer(bootstrap_servers=args.kafka_brokers)
        
        events = map(shift, tqdm(events))
        publish_kafka_events(producer, events,
                             topic_finder=lambda ev: NodeEvent.from_event_type(ev.__class__).topic,
                             sync=args.sync)
    except NoBrokersAvailable as e:
        import sys
        print(f'fails to connect to Kafka: server={args.kafka_brokers}', file=sys.stderr)
    finally:
        if producer:
            producer.close()

if __name__ == '__main__':
    main()