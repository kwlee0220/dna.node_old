from __future__ import annotations

from typing import Union, Optional, DefaultDict
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass, replace
import itertools
import pickle

import numpy as np
import cv2
from omegaconf import OmegaConf
import json
from pathlib import Path
from tqdm import tqdm

from dna import NodeId
from dna.event import NodeTrack, KafkaEvent, TrackFeature, read_json_event_file, read_topics, open_kafka_consumer
from dna.support import iterables

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', choices=['latest', 'earliest', 'none'],
                        help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument("topics", nargs='+', help="topic names")
    parser.add_argument("--stop_on_poll_timeout", action='store_true', help="stop when a poll timeout expires")
    parser.add_argument("--offsets", metavar="csv", help="camera frame offsets")
    parser.add_argument("--output", "-o", metavar="path", default=None, help="output file.")
    return parser.parse_known_args()

def read_adjusted_track_events(track_file:str, offset:int):
    def adjust_frame_index(track:NodeTrack):
        return replace(track, frame_index=track.frame_index + offset)
    return (adjust_frame_index(track) for track in read_json_event_file(track_file, NodeTrack))
    
NODE_IDS = ['etri:04', 'etri:05', 'etri:06', 'etri:07']
def main():
    args, _ = parse_args()

    if args.offsets is not None:
        offsets = [int(vstr) for vstr in args.offsets.split(',')]
    else:
        offsets = [0] * len(args.track_files)
    shift = 0 - min(offsets)
    offsets = [o + shift for o in offsets]
    
    node_offsets:dict[NodeId,int] = defaultdict(int)
    for idx, node_id in enumerate(NODE_IDS):
        node_offsets[node_id] = offsets[idx]
    
    # 주어진 topic에 저장된 모든 event를 download한다.
    print(f"downloading topic events.")
    full_events:list[KafkaEvent] = []
    download_counts:DefaultDict[str,int] = defaultdict(int)
    with closing(open_kafka_consumer(brokers=args.kafka_brokers,
                                     offset_reset=args.kafka_offset,
                                     key_deserializer=lambda k: k.decode('utf-8'))) as consumer:
        consumer.subscribe(args.topics)
        for record in tqdm(read_topics(consumer, initial_timeout_ms=10000, timeout_ms=3000,
                                       stop_on_poll_timeout=args.stop_on_poll_timeout)):
            if record.topic == 'node-tracks':
                full_events.append(NodeTrack.deserialize(record.value))
            elif record.topic == 'track-features':
                full_events.append(TrackFeature.deserialize(record.value))
            download_counts[record.topic] += 1
    print(f"downloaded {len(full_events)} topic events: {dict(download_counts)}")
                
    print(f"shifting {len(full_events)} events.")
    shifted_events:list[KafkaEvent] = []
    for node_id, offset in zip(NODE_IDS, offsets):
        node_events = [ev for ev in full_events if ev.node_id == node_id]
        frame_delta = node_events[offset].frame_index - node_events[0].frame_index
        ts_delta = node_events[offset].ts - node_events[0].ts
        
        shifted_events.extend([replace(ev, frame_index = ev.frame_index-frame_delta, ts = ev.ts-ts_delta) for ev in node_events[offset:]])
            
    shifted_events.sort(key=lambda ev:ev.ts)
    print(f"sorted {len(shifted_events)} events by timestamp.")
    
    print(f"writing: {len(shifted_events)} records into file '{args.output}'")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as fp:
        for ev in tqdm(shifted_events):
            pickle.dump(ev, fp)
    
if __name__ == '__main__':
    main()