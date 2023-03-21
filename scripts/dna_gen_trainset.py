from __future__ import annotations
from typing import List, Dict, Generator, Any, Tuple, Optional

from contextlib import closing
from datetime import timedelta
from pathlib import Path

import yaml
from omegaconf import OmegaConf
import cv2

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import dna
from dna import Point, Box, Size2d, Frame
from dna.conf import load_node_conf, get_config
from scripts.utils import load_camera_conf
from dna.camera import ImageProcessor, FrameProcessor, create_camera_from_conf
from dna.tracker import TrackState
from dna.node import EventProcessor, TrackEvent, TrackId, TrackEventPipeline
from dna.node.zone import ZoneEvent, ZonePipeline
from dna.node.utils import read_tracks_json
from dna.zone import Zone


FINE_ZONES = {
    ('CA', 'C'): 'A',
    ('CB', 'C'): 'A',
    ('AC', 'C'): 'B',
    ('BC', 'C'): 'B',
    ('AD', 'A'): 'C',
    ('AE', 'A'): 'C',
    ('DA', 'A'): 'D',
    ('EA', 'A'): 'D',
    ('DA', 'D'): 'E',
    ('AD', 'D'): 'F',
    ('CA', 'A'): 'G',
    ('BA', 'B'): 'G',
    ('BA', 'A'): 'G',
    ('BC', 'B'): 'G',
    ('AB', 'A'): 'H',
    ('AB', 'B'): 'H',
    ('CB', 'B'): 'H',
    ('AC', 'A'): 'H',

    ('AE', 'E'): 'X',
    ('EA', 'E'): 'X',
}

ZONE_SEQUENCES = {
    'etri:04': {
        'CA': ['A', 'G', 'C', 'F'],
        'CB': ['A', 'H'],
        'AC': ['E', 'D', 'B'],
        'BC': ['G', 'B'],
        'BA': ['G'],
        'AB': ['H'],
    },
    'etri:05': {
        'CA': ['A', 'C', 'F'],
        'CB': ['A', 'H'],
        'AC': ['E', 'D', 'B'],
        'BC': ['G', 'B'],
        'BA': ['G', 'C', 'F'],
        'AB': ['E', 'D', 'H'],
    },
    'etri:06': {
        'CA': ['A', 'C', 'F'],
        'CB': ['A', 'H'],
        'AC': ['E', 'D', 'B'],
        'BC': ['G', 'B'],
        'AB': ['E', 'D', 'H'],
    },
    'etri:07': {
        'AD': ['C', 'F'],
        'AE': ['C'],
        'DA': ['E', 'D'],
        'EA': ['D'],
    },
}

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate ReID training data set")
    parser.add_argument("track_video")
    parser.add_argument("track_file")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--margin", metavar="pixels", type=int, default=0, help="margin pixels for cropped image")
    parser.add_argument("--motions", metavar="path", help="path to the motion mapping file")
    parser.add_argument("--matches", metavar="path", default=None, help="path to the tracklet match file")
    parser.add_argument("--start_gidx", metavar="number", type=int, default=0, help="starting global tracjectory number")
    parser.add_argument("--output", "-o", metavar="directory path", help="output training data directory path")
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


def load_tracklets_by_frame(tracklet_gen:Generator[TrackEvent, None, None]) -> Dict[int,List[TrackEvent]]:
    tracklets:Dict[int,List[TrackEvent]] = dict()
    for track in tracklet_gen:
        tracks = tracklets.get(track.frame_index)
        if tracks is None:
            tracklets[track.frame_index] = [track]
        else:
            tracklets[track.frame_index].append(track)

    return tracklets


def to_crop_file(dir:Path, track_id:str, ts:int) -> Path:
    return dir / track_id.rjust(5, '0') / f'{ts}.png'


class TrackletCropWriter(FrameProcessor):
    def __init__(self, tracks_per_frame_index:Dict[int, List[TrackEvent]],
                 zone_events:Dict[TrackId,List[ZoneEvent]],
                 motions: Dict[TrackId,Tuple[str,str]],
                 global_tracklet_mappings: Dict[TrackId,Tuple[str,str,str]],
                 output_dir:str,
                 margin:int=5,
                 min_size:Size2d=Size2d(20, 20)) -> None:
        self.tracks_per_frame_index = tracks_per_frame_index
        self.zone_events = zone_events
        self.motions = motions
        self.global_tracklet_mappings = global_tracklet_mappings
        self.margin = margin
        self.min_size = min_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        self.empty_motions = set()
        self.non_global_tracklets = set()
        
    def on_started(self, proc:ImageProcessor) -> None:
        pass
    def on_stopped(self) -> None:
        pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        tracks = self.tracks_per_frame_index.pop(frame.index, None)
        if tracks is None:
            return frame
        
        for track in tracks:
            if track.state == TrackState.Deleted:
                self.sessions.pop(track.track_id, None)
            elif track.state == TrackState.Confirmed or track.state == TrackState.Tentative:
                crop_file = self.crop_file_path(track)
                if crop_file:
                    crop_file.parent.mkdir(parents=True, exist_ok=True)

                    h, w, _ = frame.image.shape
                    border = Box.from_size((w,h))

                    crop_box = track.location.expand(self.margin).to_rint()
                    crop_box = crop_box.intersection(border)
                    if crop_box.size() > self.min_size:
                        track_crop = crop_box.crop(frame.image)
                        cv2.imwrite(str(crop_file), track_crop)

        return frame

    def crop_file_path(self, track:TrackEvent) -> str:
        motion = self.motions.get(track.track_id)
        if not motion:
            if (track.node_id, track.track_id) not in self.empty_motions:
                print(f"cannot find motion: {track.node_id}, {track.track_id}")
                self.empty_motions.add((track.node_id, track.track_id))
            return None
        motion = ''.join(motion)

        gid = self.global_tracklet_mappings.get((track.node_id, track.track_id))
        if not gid:
            if (track.node_id, track.track_id) not in self.non_global_tracklets:
                print(f'cannot find global track: {track.node_id}, {track.track_id}')
                self.non_global_tracklets.add((track.node_id, track.track_id))
            return None
        
        seqs = ZONE_SEQUENCES[track.node_id]
        seq = seqs.get(motion)
        if not seq:
            print(f"unknown motion: node={track.node_id}, track={track.track_id}, motion={motion}")
            return None
        
        zone_events = self.zone_events.get(track.track_id)
        if not motion:
            print(f"unknown zone event: track={track.track_id}")
            return None
        
        zev_idx = next(iter(idx for idx, ze in enumerate(zone_events) if ze.frame_index == track.frame_index), None)
        if zev_idx is None:
            return None
        zone_ev = zone_events.pop(zev_idx)

        if zone_ev.is_inside() or zone_ev.is_entered():
            key = (motion, zone_ev.zone_id)
            fine_zone = FINE_ZONES.get(key)
            if fine_zone and fine_zone != 'X':
                if fine_zone in seq:
                    part_no = seq.index(fine_zone)
                    node_id = track.node_id[track.node_id.index(':')+1:]
                    return self.output_dir / gid / f'{part_no:02d}_{fine_zone}_{node_id}_{track.ts}.png'
                else:
                    return None
            elif fine_zone == 'X':
                return None
            else:
                print(f'unknown fine_zone key: {key}')
                return None
        else:
            return None

def load_motions(motion_file:str) -> Dict[TrackId,Tuple[str,str]]:
    with open(motion_file, 'r') as fp:
        csv_lines = (tuple(line.rstrip().split(',')) for line in fp.readlines())
        return {track_id:(entry, exit) for track_id, entry, exit in csv_lines}

def load_tracklet_matches(file:str, start_index:int=0) -> Dict[(str, TrackId),str]:
    global_tracklet_mappings:Dict[(str, TrackId),str] = dict()
    with open(file, 'r') as fp:
        for idx, line in enumerate(fp.readlines(), start=start_index):
            gid, t1, t2, t3, t4 = tuple([f'g{idx:05d}'] + line.rstrip().split(','))
            if t1 != 'X':
                global_tracklet_mappings[('etri:04', t1)] = gid
            if t2 != 'X':
                global_tracklet_mappings[('etri:05', t2)] = gid
            if t3 != 'X':
                global_tracklet_mappings[('etri:06', t3)] = gid
            if t4 != 'X':
                global_tracklet_mappings[('etri:07', t4)] = gid
        return global_tracklet_mappings
    

class CollectZoneInfo(EventProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.zone_events:Dict[TrackId,List[ZoneEvent]] = dict()

    def handle_event(self, ev:Any) -> None:
        if isinstance(ev, ZoneEvent):
            zone_events = self.zone_events.get(ev.track_id)
            if not zone_events:
                zone_events:List[ZoneEvent] = []
                self.zone_events[ev.track_id] = zone_events
            zone_events.append(ev)


def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf(args, ['show_progress'])
        
    tracks = (track for track in read_tracks_json(args.track_file) if not track.is_deleted())
    tracks_per_frame_index = load_tracklets_by_frame(tracks)
        
    motion_file = OmegaConf.select(args_conf, 'motions', default=None)
    motions = load_motions(motion_file) if motion_file else None

    output_dir = Path(args.output)
        
    tracklet_match_file = args.matches
    global_tracklet_mappings = load_tracklet_matches(tracklet_match_file, args.start_gidx) if tracklet_match_file else None

    publishing_conf = OmegaConf.select(conf, 'publishing')
    track_pipeline = TrackEventPipeline(conf.id, publishing_conf)
    zone_pipeline:ZonePipeline = track_pipeline.plugins.get('zone_pipeline')

    node_dir = Path(args.output)
    collector = CollectZoneInfo()
    zone_pipeline.event_queues['zone_events'].add_listener(collector)
    
    for track_ev in read_tracks_json(args.track_file):
        track_pipeline.input_queue.publish_event(track_ev)
    track_pipeline.close()
    
    # 카메라 설정 정보 추가
    conf.camera = {'uri': args.track_video, 'sync': False, }
    camera = create_camera_from_conf(conf.camera)
    
    img_proc = ImageProcessor(camera.open(), conf)
    training_data_writer = TrackletCropWriter(tracks_per_frame_index, collector.zone_events, motions,
                                              global_tracklet_mappings, args.output, margin=args.margin)
    img_proc.add_frame_processor(training_data_writer)
    result: ImageProcessor.Result = img_proc.run()

if __name__ == '__main__':
	main()