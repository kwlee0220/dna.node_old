from __future__ import annotations

from typing import Tuple
from collections.abc import Generator
import dataclasses
from contextlib import closing
from tqdm import tqdm
from pathlib import Path
import argparse

from dna import initialize_logger
from dna.event import KafkaEvent, NodeTrack, open_kafka_producer, read_event_file, publish_kafka_events
from dna.node import NodeEventType
from scripts import *


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Replay Kafka events")
    
    parser.add_argument("file", help="events file (json or pickle format)")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'],
                        help="Kafka broker hosts list")
    parser.add_argument("--topic", help="target topic name")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    print(f"uploading events to topic '{args.topic}' from the file '{args.file}'.")
    with closing(open_kafka_producer(args.kafka_brokers)) as producer:
        for kv in read_event_file(args.file, event_type=NodeTrack):
            if isinstance(kv, tuple):
                producer.send(args.topic, value=kv[1], key=kv[0])
            else:
                producer.send(args.topic, value=kv.serialize(), key=kv.key())

  
# def read_event_file(file:str) -> Generator[tuple[object,object], None, None]:
#     import pickle
#     from pathlib import Path
    
#     with open(file, 'rb') as fp:
#         try:
#             while True:
#                 yield pickle.load(fp)
#         except EOFError:
#             return

if __name__ == '__main__':
    main()