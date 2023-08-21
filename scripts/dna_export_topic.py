from __future__ import annotations

import argparse
import pickle
from tqdm import tqdm
from contextlib import closing
from pathlib import Path
import time

from kafka import KafkaConsumer

from dna import initialize_logger
from dna.event import open_kafka_consumer, read_topics
from dna.node import NodeEventType
from scripts import *


def parse_args():
    # parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    parser = argparse.ArgumentParser(description="Tracklet and tracks commands")
    
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--topic", help="target topic name")
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")
    parser.add_argument("--stop_on_poll_timeout", action='store_true', help="stop when a poll timeout expires")
    parser.add_argument("--sleep_millis", metavar="millis-seconds", type=int, default=0, help="sleep milli-seconcs for each file write.")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    with closing(open_kafka_consumer(args.kafka_brokers, args.kafka_offset)) as consumer:
        consumer.subscribe([args.topic])
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as fp:
            print(f"reading events from the topics '{args.topic}' and write to '{args.output}'.")
            for record in tqdm(read_topics(consumer, initial_timeout_ms=10000, timeout_ms=3000,
                                           stop_on_poll_timeout=args.stop_on_poll_timeout)):
                pickle.dump((record.key, record.value), fp)
                if args.sleep_millis:
                    time.sleep(args.sleep_millis / 1000)

if __name__ == '__main__':
    main()