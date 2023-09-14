from __future__ import annotations

from typing import Tuple
from contextlib import closing
import argparse

from kafka import KafkaProducer

from dna.event import open_kafka_producer, read_event_file, process_kafka_events, KafkaEvent, Timestamped, TrackFeature
from dna.event.utils import read_event_file
from dna.node import NodeEventType
from scripts import *


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Replay Kafka events")
    
    parser.add_argument("file", help="event file (pickle format)")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--show_progress", action='store_true', default=False)
    parser.add_argument("--sync", action='store_true', default=False)

    return parser.parse_known_args()
    
def main():
    args, _ = parse_args()
    
    print(f"publish events from the file '{args.file}'.")
    with closing(open_kafka_producer(args.kafka_brokers)) as producer:
        def publish_event(ev:Timestamped) -> None:
            topic = NodeEventType.find_topic(ev)
            producer.send(topic, value=ev.serialize(), key=ev.key())
            
        process_kafka_events(read_event_file(args.file), publish_event, sync=args.sync,
                             show_progress=args.show_progress, max_wait_ms=1000)
        # process_kafka_events(read_event_file(args.file), print_event)

if __name__ == '__main__':
    main()