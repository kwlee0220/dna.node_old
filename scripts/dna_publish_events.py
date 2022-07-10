
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
    parser.add_argument("log_path", help="configuration file path")
    parser.add_argument("--server", help="bootstrap-servers", default='localhost:9092')
    parser.add_argument("--topic", help="topic name", default='node-tracks')
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()

    producer = KafkaProducer(bootstrap_servers=[args.server])

    heap = []
    heapq.heapify(heap)

    with open(args.log_path, 'r') as fp:
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
                if last_ts > 0:
                    remains = track.ts - last_ts
                    if remains > 30:
                        time.sleep(remains / 1000.0)
                producer.send(args.topic, value=track.serialize(), key=track.key())
                producer.flush()
                last_ts = track.ts
    producer.close()

if __name__ == '__main__':
	main()