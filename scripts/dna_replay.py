
from typing import Dict, Any

# from contextlib import closing
from collections import defaultdict
from pathlib import Path
import pickle
from contextlib import suppress
import time

from kafka import KafkaConsumer, KafkaProducer
from omegaconf import OmegaConf
from argparse import Namespace
import numpy as np

from dna import initialize_logger, config
from dna.node import NodeId, TrackFeature, TrackEvent, KafkaEvent
from dna.node.zone import TrackletMotion

TOPIC_TRACK_EVENTS = "track-events"
TOPIC_MOTIONS = "track-motions"
TOPIC_FEATURES = 'track-features'


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("files", nargs='+', help="kafka server")
    parser.add_argument("--bootstrap_servers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--sync", action='store_true', default=False)
    parser.add_argument("--logger", metavar="file path", default=None, help="logger configuration file path")

    return parser.parse_known_args()
            
def topic_for(ev):
    if isinstance(ev, TrackEvent):
        return TOPIC_TRACK_EVENTS
    elif isinstance(ev, TrackFeature):
        return TOPIC_FEATURES
    elif isinstance(ev, TrackletMotion):
        return TOPIC_MOTIONS
    else:
        raise AssertionError()

def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    with open(args.files[0], 'rb') as fp: events = pickle.load(fp)
    with open(args.files[1], 'rb') as fp: motions = pickle.load(fp)
    with open(args.files[2], 'rb') as fp: features = pickle.load(fp)
    collection = events + motions + features
    collection.sort(key=lambda v: v.ts)
    
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    
    now = round(time.time() * 1000)
    delta = now - collection[0].ts
    
    for ev in collection:
        topic = topic_for(ev)
        
        if args.sync:
            now = round(time.time() * 1000)
            now_expected = ev.ts + delta
            sleep_ms = now_expected - now
            if sleep_ms > 20:
                if sleep_ms > 5 * 1000:
                    print(f"sleep: {sleep_ms/1000:.3f}s")
                time.sleep((sleep_ms)/1000)
        
        producer.send(topic, value=ev.serialize(), key=ev.key().encode('utf-8'))
    producer.close()

if __name__ == '__main__':
    main()