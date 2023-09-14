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
from dna.event import NodeTrack, KafkaEvent, TrackFeature, read_json_event_file, read_topics, open_kafka_consumer, Timestamped, process_kafka_events
from dna.node import NodeEventType
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

def calibrate(ev:KafkaEvent, offset_delta:int, ts_delta:int) -> KafkaEvent:
    if hasattr(ev, 'first_ts'):
        return replace(ev, frame_index=ev.frame_index-offset_delta, first_ts=ev.first_ts-ts_delta, ts=ev.ts-ts_delta)
    else:
        return replace(ev, frame_index=ev.frame_index-offset_delta, ts=ev.ts-ts_delta)


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
            type = NodeEventType.from_topic(record.topic)
            full_events.append(type.deserialize(record.value))
            download_counts[record.topic] += 1
    print(f"downloaded {len(full_events)} topic events: {dict(download_counts)}")
                
    print(f"shifting {len(full_events)} events.")
    
    # node별로 전체 event를 grouping한다.
    node_event_groups:dict[NodeId,list[KafkaEvent]] = iterables.groupby(full_events, lambda ev: ev.node_id)
    
    shifted_events:list[KafkaEvent] = []
    for node_id, offset in zip(NODE_IDS, offsets):
        node_events:list[KafkaEvent] = node_event_groups.get(node_id)
        if node_events is None:
            continue
        
        # 각 노드별로 frame_index의 offset에 따른 timestamp의 delta를 구한다.
        # 이때는 거의 모든 frame별로 event가 생성되는 NodeTrack 이벤트를 사용한다.
        node_tracks:list[NodeTrack] = [ev for ev in node_events if isinstance(ev, NodeTrack)]
        if len(node_tracks) > 0:
            # 첫번째 event의 frame_index를 기준으로 삼아, 이것보다 offset만큼 큰 frame_index를 갖는 event를 찾아서
            # 두 event 사이의 timestamp 차이를 구한다.
            first_event = node_tracks[0]
            idx, found = iterables.find_first(node_tracks, lambda ev: (ev.frame_index - first_event.frame_index) == offset)
            ts_delta = found.ts - first_event.ts
            
            # 계산된 frame_index offset과 timestamp offset을 이용하여 지정된 node에서 생성된 모든 event들을 calibration한다.
            # 'NodeTrack'의 경우에는 first_ts가 존재하기 때문에 이것도 함께 보정해야 한다.
            if offset > 0:
                shifted_events.extend([calibrate(ev, offset_delta=offset, ts_delta=ts_delta)
                                        for ev in node_events if ev.frame_index >= offset])
            else:
                shifted_events.extend(node_events)
            
    shifted_events.sort(key=lambda ev:ev.ts)
    print(f"sorted {len(shifted_events)} events by timestamp.")
    
    print(f"writing: {len(shifted_events)} records into file '{args.output}'")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as fp:
        for ev in tqdm(shifted_events):
            pickle.dump(ev, fp)

    
    
if __name__ == '__main__':
    main()