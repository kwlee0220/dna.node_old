
import heapq

import time
import numpy as np
from kafka import KafkaProducer, KafkaConsumer

from dna import Box
from dna.tracker import TrackState, Track
from dna.node import TrackEvent


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("log_paths", nargs='+', help="configuration file path")
    parser.add_argument("--servers", help="bootstrap-servers", default='kafka01:9092,kafka02:9092,kafka03:9092')
    parser.add_argument("--topic", help="topic name", default='node-tracks')
    parser.add_argument("--sync", help="sync to publish events", action='store_true')
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()

    producer = KafkaProducer(bootstrap_servers=args.servers.split(','))
    for log_path in args.log_paths:
        heap = []
        heapq.heapify(heap)

        with open(log_path, 'r') as fp:
            last_ts = 0
            while True:
                line = fp.readline().rstrip()
                if len(line) > 0:
                    te = TrackEvent.from_json(line)
                    heapq.heappush(heap, te)
                elif len(heap) == 0:
                    break
                
                if len(heap) >= 32 or len(line) == 0:
                    track: TrackEvent = heapq.heappop(heap)
                    if args.sync and last_ts > 0:
                        remains = track.ts - last_ts
                        if remains > 30:
                            time.sleep(remains / 1000.0)
                    producer.send(args.topic, value=track.serialize(), key=track.key())
                    producer.flush()
                    last_ts = track.ts
    producer.close()

if __name__ == '__main__':
	main()