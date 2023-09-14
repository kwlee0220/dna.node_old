from __future__ import annotations

import argparse
from io import TextIOWrapper
import pickle
from tqdm import tqdm
from contextlib import closing
from pathlib import Path
import time

from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord

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
    parser.add_argument("--json", action='store_true', help="export events in json format.")
    parser.add_argument("--sleep_millis", metavar="millis-seconds", type=int, default=0, help="sleep milli-seconcs for each file write.")

    return parser.parse_known_args()


class ConsumerRecordPickleWriter:
    def __init__(self, path:Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp:TextIOWrapper = open(path, 'wb')
        
    def close(self) -> None:
        if self.fp is not None:
            self.fp.close()
            self.fp = None
            
    def write(self, record:ConsumerRecord) -> None:
        pickle.dump((record.key, record.value), self.fp)
        self.fp.flush()
        

class ConsumerRecordTextWriter:
    def __init__(self, path:Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp:TextIOWrapper = open(path, 'w')
        
    def close(self) -> None:
        if ( self.fp is not None ):
            self.fp.close()
            self.fp = None
            
    def write(self, record:ConsumerRecord) -> None:
        json = record.value.decode('utf-8')
        self.fp.write(json + '\n')
        
            

def main():
    args, _ = parse_args()
    
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    with closing(open_kafka_consumer(args.kafka_brokers, args.kafka_offset)) as consumer:
        consumer.subscribe([args.topic])
        
        path = Path(args.output)
        suffix = path.suffix
        writer = None
        if suffix == '.pickle':
            writer = ConsumerRecordPickleWriter(path)
        elif suffix == '.json':
            writer = ConsumerRecordTextWriter(path)
        else:
            raise ValueError(f"unsupported extension: '{suffix}'")
        
        with closing(writer) as writer:
            print(f"reading events from the topics '{args.topic}' and write to '{args.output}'.")
            records = read_topics(consumer, initial_timeout_ms=10000, timeout_ms=3000,
                                  stop_on_poll_timeout=args.stop_on_poll_timeout)
            for record in tqdm(records, desc='exporting records'):
                writer.write(record)
                if args.sleep_millis:
                    time.sleep(args.sleep_millis / 1000)
        
        # Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # with open(args.output, 'wb') as fp:
        #     print(f"reading events from the topics '{args.topic}' and write to '{args.output}'.")
        #     records = read_topics(consumer, initial_timeout_ms=10000, timeout_ms=3000,
        #                             stop_on_poll_timeout=args.stop_on_poll_timeout)
        #     for record in tqdm(records, desc='exporting records'):
        #         pickle.dump((record.key, record.value), fp)
        #         fp.flush()
        #         if args.sleep_millis:
        #             time.sleep(args.sleep_millis / 1000)

if __name__ == '__main__':
    main()