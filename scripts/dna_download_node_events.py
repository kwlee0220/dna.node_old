from __future__ import annotations

import argparse
import pickle
from tqdm import tqdm
from contextlib import closing
from pathlib import Path

from kafka import KafkaConsumer

from dna import initialize_logger
from dna.event import open_kafka_consumer, download_topics
from dna.node import NodeEvent
from scripts import *


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='latest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    with closing(open_kafka_consumer(args.kafka_brokers, args.kafka_offset, key_deserializer=lambda k: k.decode('utf-8'))) as consumer:
        consumer.subscribe(NodeEvent.topics())
        
        print(f"reading events from the topics '{NodeEvent.topics()}'.")
        full_events = download_topics(consumer, topics=NodeEvent.topics(),
                                      deserializer_func=lambda t: NodeEvent.from_topic(t).deserializer,
                                      timeout_ms=2000)
        full_events = list(tqdm(full_events))
        
        max_track_id = max(int(ev.track_id) for ev in full_events)
        max_ts = max(ev.ts for ev in full_events)
        max_frame_index = max(ev.frame_index for ev in full_events)
        
        print(f"sorting {len(full_events)} events by timestamp.")
        full_events.sort(key=lambda ev:ev.ts)
        
        print(f"writing: {len(full_events)} records into file '{args.output}'")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'wb') as fp:
            for ev in full_events:
                pickle.dump(ev, fp)
        consumer.close()
        
        print(f'max track_id: {max_track_id}')
        print(f'max frame_index: {max_frame_index}')
        print(f'max ts: {max_ts}')

if __name__ == '__main__':
    main()