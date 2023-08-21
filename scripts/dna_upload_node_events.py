from __future__ import annotations

from typing import Tuple
from contextlib import closing
from tqdm import tqdm
import argparse

from dna.event import open_kafka_producer, read_event_file, publish_kafka_events
from dna.event.utils import read_event_file
from dna.node import NodeEventType
from scripts import *


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Replay Kafka events")
    
    parser.add_argument("file", help="event file (pickle format)")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--sync", action='store_true', default=False)

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    
    print(f"publish events from the file '{args.file}'.")
    with closing(open_kafka_producer(args.kafka_brokers)) as producer:
        publish_kafka_events(producer, read_event_file(args.file), topic_finder=NodeEventType.find_topic, sync=args.sync)

if __name__ == '__main__':
    main()