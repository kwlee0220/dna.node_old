
from typing import Dict, Any

import time
import pickle
from tqdm import tqdm

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from dna import initialize_logger
from dna.event import TrackEvent, TrackFeature, TrackletMotion
from scripts import update_namespace_with_environ

TOPIC_TRACK_EVENTS = "track-events"
TOPIC_MOTIONS = "track-motions"
TOPIC_FEATURES = 'track-features'


import argparse
def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="Tracklet and tracks commands")
    
    parser.add_argument("files", nargs='+', help="event pickle files")
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
    args = update_namespace_with_environ(args)

    initialize_logger(args.logger)
    
    events = []
    for file in args.files:
        with open(file, 'rb') as fp:
            events.extend(pickle.load(fp))
    events.sort(key=lambda v: v.ts)
    
    try:
        producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    
        now = round(time.time() * 1000)
        delta = now - events[0].ts
        
        for ev in tqdm(events):
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
    except NoBrokersAvailable as e:
        import sys
        print(f'fails to connect to Kafka: server={args.bootstrap_servers}', file=sys.stderr)

if __name__ == '__main__':
    main()