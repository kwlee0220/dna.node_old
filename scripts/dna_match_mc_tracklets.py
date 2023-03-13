
from __future__ import annotations

import sys
from typing import List, Dict, Tuple, Generator, Set
from dataclasses import replace
import itertools

from dna import  initialize_logger
from dna.tracker import TrackState
from dna.node import TrackEvent, Tracklet
from dna.tracklet.tracklet_matcher import match_tracklets
from dna.support.load_tracklets import read_tracks_json

import logging
LOGGER = logging.getLogger('dna.node.sync_frames')

def load_tracklets(track_file:str, offset:int, max_camera_dist:float) -> Tuple[str, Dict[int,Tracklet]]:
    def is_valid_track(ev:TrackEvent) -> bool:
        if ev.is_deleted():
            return True
        elif ev.state != TrackState.Confirmed and ev.state != TrackState.Tentative:
            return False
        return ev.distance and ev.distance <= max_camera_dist and ev.world_coord

    node_id = ''
    tracklets:Dict[int,Tracklet] = dict()
    for te in read_tracks_json(track_file):
        # 일부 TrackEvent의 "world_coord"와 "distance" 필드가 None 값을 갖기 때문에
        # 해당 event는 제외시킨다. 또한 대상 물체와 카메라와의 거리 (distance)가
        # 일정거리('max_camera_dist')보다 큰 경우도 추정된 실세계 좌표('woorld_dist') 값에
        # 정확도에 떨어져서 제외시킨다.
        if not is_valid_track(te):
            continue
        
        tracklet = tracklets.get(te.track_id)
        if tracklet is None:
            tracklet = Tracklet(te.track_id, [])
            tracklets[te.track_id] = tracklet
            
        if offset != 0:
            te = replace(te, frame_index=te.frame_index + offset)
        tracklet.append(te)
        node_id = te.node_id
    return node_id, tracklets


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Synchronize two videos")
    parser.add_argument("track_files", nargs='+', help="track json files")
    parser.add_argument("--max_camera_distance", type=float, metavar="meter", default=50,
                        help="max. distance from camera (default: 55)")
    parser.add_argument("--tracklet_overlap", type=int, metavar="count", default=10, 
                        help="maximum frame difference between videos (default: 10)")
    parser.add_argument("--track_distance", type=float, metavar="value", default=10, 
                        help="maximum valid distance between tracks (default: 10)")
    parser.add_argument("--tracklet_distance", type=float, metavar="value", default=7, 
                        help="maximum valid average distance for matching tracklets (default: 7)")
    parser.add_argument("--offsets", metavar="csv", help="camera frame offsets")
    parser.add_argument("--output", "-o", metavar="dir", help="output directory.", default=None)
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def find_by_index(full_matches, index:int, track_id:int):
    for idx, full_match in enumerate(full_matches):
        if full_match[index] == track_id:
            return idx, full_match
    return -1, None

def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    
    if args.offsets is not None:
        offsets = [max(0, int(vstr)) for vstr in args.offsets.split(',')]
    else:
        offsets = [0] * len(args.video_uris)
    
    full_matches = []
    nodes = [load_tracklets(track_file, -offset, args.max_camera_distance)
                for track_file, offset in zip(args.track_files, offsets)]
    node_indexes = { node[0]:idx for idx, node in enumerate(nodes) }
    
    for node1, node2 in itertools.combinations(nodes, 2):
        pos1 = node_indexes[node1[0]]
        pos2 = node_indexes[node2[0]]
        matches, _ = match_tracklets(node1[1], node2[1],
                                     ignore_matches=set(),
                                     min_overlap_length=args.tracklet_overlap,
                                     max_track_distance=args.track_distance,
                                     max_tracklet_distance=args.tracklet_distance)
        for match, dist in matches.items():
            idx, full_match = find_by_index(full_matches, pos1, match[0])
            if idx >= 0:
                full_match[pos1] = match[0]
                full_match[pos2] = match[1]
            else:
                full_match = [-1] * len(nodes)
                full_match[pos1] = match[0]
                full_match[pos2] = match[1]
                full_matches.append(full_match)
                
    for fm in full_matches:
        print(fm)

if __name__ == '__main__':
    main()