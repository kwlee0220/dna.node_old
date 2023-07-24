from __future__ import annotations

from typing import Union, Optional
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import Box, Image, color, Frame, Point, TrackletId, initialize_logger, config, Size2d
from dna.camera import Camera
from dna.camera.video_writer import VideoWriter
from dna.event import NodeTrack, read_topics
from dna.node import stabilizer
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.support import plot_utils
from dna.track import TrackState
from dna.assoc import GlobalTrack
from scripts import update_namespace_with_environ

COLORS = {
    'etri:04': color.BLUE,
    'etri:05': color.GREEN,
    'etri:06': color.YELLOW,
    'etri:07': color.INDIGO,
    'global': color.RED
}

RADIUS_GLOBAL = 15
RADIUS_LOCAL = 7

MAX_DISTS = {'etri:04': 45, 'etri:05': 45, 'etri:06': 45, 'etri:07': 45 }
LOCALIZERS:dict[str,WorldCoordinateLocalizer] = dict()
LOCALIZER = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                     camera_index=0, contact_point=ContactPointType.Simulation)

@dataclass(frozen=True)
class Location:
    luid: int
    point: Point
    distance: float

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", default=['localhost:9092'], help="Kafka broker hosts list")
    parser.add_argument("--kafka_offset", default='earliest', help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


class GlobalTrackDrawer:
    def __init__(self, title:str, localizer:WorldCoordinateLocalizer, world_image:Image,
                 *,
                 output_video:str=None) -> None:
        self.title = title
        self.localizer = localizer
        self.world_image = world_image
        cv2.namedWindow(self.title)
        
        if output_video:
            self.writer = VideoWriter(Path(output_video).resolve(), 10, Size2d.from_image(world_image))
            self.writer.open()
        else:
            self.writer = None
        
    def close(self) -> None:
        if self.writer:
            self.writer.close()
        cv2.destroyWindow(self.title)
    
    def draw_tracks(self, gtracks:list[GlobalTrack]) -> Image:
        convas = self.world_image.copy()
        
        ts = max((gl.ts for gl in gtracks), default=None)
        convas = cv2.putText(convas, f'ts={ts}',
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
    
        for gtrack in gtracks:
            gloc = self.to_image_coord(gtrack.location)
            convas = cv2.circle(convas, gloc, radius=RADIUS_GLOBAL, color=color.RED, thickness=-1, lineType=cv2.LINE_AA)
            
            for ltrack in gtrack.support:
                track_color = COLORS[ltrack.node]
                sample = self.to_image_coord(ltrack.location)
                convas = cv2.line(convas, gloc, sample, track_color, thickness=1, lineType=cv2.LINE_AA)
                convas = cv2.circle(convas, sample, radius=RADIUS_LOCAL, color=track_color, thickness=-1, lineType=cv2.LINE_AA)
            
        # convas = ROI.crop(convas)
        if self.writer:
            self.writer.write(convas)
        cv2.imshow(self.title, convas)
        
        return convas
        
    def to_image_coord(self, world_coord:Point) -> tuple[float,float]:
        pt_m = self.localizer.from_world_coord(world_coord.xy)
        return tuple(Point(self.localizer.to_image_coord(pt_m)).to_rint().xy)


def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    world_image = cv2.imread("regions/etri_testbed/ETRI_221011.png", cv2.IMREAD_COLOR)
    localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                         camera_index=0, contact_point=ContactPointType.Simulation)
    drawer = GlobalTrackDrawer(title="Multiple Objects Tracking", localizer=localizer, world_image=world_image)
    
    consumer = KafkaConsumer(bootstrap_servers=args.kafka_brokers,
                            #  group_id='draw_global_tracks',
                            #  enable_auto_commit=False,
                             auto_offset_reset=args.kafka_offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['global-tracks'])
    
    done = False
    last_ts = -1
    tracks:list[GlobalTrack] = []
    while not done:
        for record in read_topics(consumer, timeout_ms=500):
            track = GlobalTrack.deserialize(record.value)
            print(track)
            
            if last_ts < 0:
                last_ts = track.ts
                
            if last_ts != track.ts:
                drawer.draw_tracks(tracks)
                
                # delay_ms = (track.ts - current_ts)
                delay_ms = track.ts - last_ts
                if delay_ms <= 0:
                    print(f"delay={delay_ms}")
                    delay_ms = 1
                elif delay_ms >= 5000:
                    print(f"delay={delay_ms}")
                    delay_ms = 100
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    done = True
                    break
                    
                tracks.clear()
                last_ts = track.ts
                
            tracks.append(track)
    drawer.close()

if __name__ == '__main__':
    main()