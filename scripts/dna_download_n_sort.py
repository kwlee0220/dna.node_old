
from typing import List

import pickle
from contextlib import suppress
from kafka import KafkaConsumer
import argparse

from dna import initialize_logger
from dna.node import TrackFeature, TrackEvent
from dna.node.zone import TrackletMotion

TOPIC_TRACK_EVENTS = "track-events"
TOPIC_MOTIONS = "track-motions"
TOPIC_FEATURES = 'track-features'


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("--bootstrap_servers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--auto_offset_reset", default='earliest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()


def download(consumer:KafkaConsumer, topic:str, file:str, deserializer):
    consumer.subscribe(topic)
    
    collection = []
    while True:
        partitions = consumer.poll(timeout_ms=1000, max_records=100)
        if partitions:
            for partition in partitions.values():
                for serialized in partition:
                    collection.append(deserializer(serialized.value))
        else:
            break
        
    print(f"{topic}: loading {len(collection)} records")
    collection.sort(key=lambda v: v.ts)
    with open(file, 'wb') as fp:
        pickle.dump(collection, fp)


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    consumer = KafkaConsumer(bootstrap_servers=args.bootstrap_servers,
                             auto_offset_reset=args.auto_offset_reset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    with suppress(EOFError): download(consumer, TOPIC_TRACK_EVENTS, f"output/{TOPIC_TRACK_EVENTS}.pickle", TrackEvent.deserialize)
    with suppress(EOFError): download(consumer, TOPIC_MOTIONS, f"output/{TOPIC_MOTIONS}.pickle", TrackletMotion.deserialize)
    with suppress(EOFError): download(consumer, TOPIC_FEATURES, f"output/{TOPIC_FEATURES}.pickle", TrackFeature.deserialize)
    consumer.close()

if __name__ == '__main__':
    main()