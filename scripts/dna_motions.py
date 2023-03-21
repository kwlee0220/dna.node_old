
from typing import Tuple, List, Dict, Union, Optional
from contextlib import closing
from collections import defaultdict

import numpy as np
import cv2
from omegaconf import OmegaConf

from dna import Box, Image, BGR, color, Frame, Point, initialize_logger
from dna.conf import load_node_conf
from dna.node import EventQueue, EventListener, TrackEventPipeline
from dna.node.tracklet import Tracklet, read_tracklets
from dna.node.event_processors import PrintEvent
from dna.node.zone import Motion
from dna.node.zone.zone_pipeline import ZonePipeline
from dna.node.utils import read_tracks_json
from dna.support.text_line_writer import TextLineWriter


class MotionEventWriter(EventListener):
    def __init__(self, file_path:str) -> None:
        self.writer = TextLineWriter(file_path)

    def close(self) -> None:
        self.writer.close()
    
    def handle_event(self, motion:Motion) -> None:
        enter_id = motion.id[0]
        exit_id = motion.id[1]
        self.writer.write(f"{motion.track_id},{enter_id},{exit_id}\n")


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Draw paths")
    parser.add_argument("track_file")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--output", "-o", metavar="file path", default='stdout', help="output motion file path")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf(args)

    queue = EventQueue()
    node_id = conf.id
    tracklets = read_tracklets(read_tracks_json(args.track_file))

    publishing_conf = OmegaConf.select(conf, 'publishing')
    track_pipeline = TrackEventPipeline(node_id, publishing_conf)
    zone_pipeline:ZonePipeline = track_pipeline.plugins.get('zone_pipeline')
    
    motions = zone_pipeline.services.get('motions')
    if not motions:
        from dna.node.zone.motion_detector import MotionDetector
        motion_conf = OmegaConf.select(conf, 'motions')
        if not motion_conf:
            raise ValueError("'motion' configuration is not found")
        motions = MotionDetector(motion_conf, zone_pipeline.LOGGER.getChild('motions'))
        zone_pipeline.event_queues['last_zone_sequences'].add_listener(motions)
    motions.add_listener(MotionEventWriter(args_conf.output))

    for tracklet in tracklets.values():
        for track in tracklet:
            track_pipeline.input_queue.publish_event(track)
    track_pipeline.close()

if __name__ == '__main__':
    main()