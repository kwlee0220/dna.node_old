from __future__ import annotations

from typing import Union, Optional
from contextlib import closing
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
import time
import logging
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import Box, Size2d, Image, BGR, color, Frame, Point, TrackletId, initialize_logger, config
from dna.camera import Camera
from dna.camera.video_writer import VideoWriter
from dna.event import NodeTrack, read_topics, read_pickle_event_file
from dna.node import stabilizer
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.support import plot_utils
from dna.track import TrackState
from dna.assoc.global_locator import GlobalLocation, GlobalTrackLocator, ValidCameraDistanceRange
from scripts import update_namespace_with_environ

RADIUS_GLOBAL = 15
RADIUS_LOCAL = 7

ROI = Box.from_points(Point((30, 150)), Point((999, 1159)))

COLORS = {
    'etri:04': color.BLUE,
    'etri:05': color.GREEN,
    'etri:06': color.YELLOW,
    'etri:07': color.INDIGO,
    'global': color.RED
}
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
    parser.add_argument("file", help="events file (pickle format)")
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--start", default=1, type=int, help="start frame index")
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


class GlobalTrackDrawer:
    def __init__(self, title:str, localizer:WorldCoordinateLocalizer, world_image:Image,
                 *,
                 output_video:str) -> None:
        self.title = title
        self.localizer = localizer
        self.world_image = world_image
        cv2.namedWindow(self.title)
        
        if output_video:
            self.writer = VideoWriter(Path(output_video).resolve(), 10, Size2d.from_image(world_image))
            self.writer.open()
        
    def close(self) -> None:
        if self.writer:
            self.writer.close()
        cv2.destroyWindow(self.title)
    
    def draw_tracks(self, glocs:list[GlobalLocation]) -> Image:
        convas = self.world_image.copy()
        
        frame_index = max((gl.frame_index for gl in glocs), default=None)
        convas = cv2.putText(convas, f'frames={frame_index}',
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
    
        for gloc in glocs:
            convas = self.draw_location(convas, gloc)
            
        # convas = ROI.crop(convas)
        if self.writer:
            self.writer.write(convas)
        cv2.imshow(self.title, convas)
        
        return convas
    
    def draw_location(self, convas:Image, gloc:GlobalLocation) -> Image:
        mean = self.to_image_coord(gloc.mean)
        convas = cv2.circle(convas, mean, radius=RADIUS_GLOBAL, color=color.RED, thickness=-1, lineType=cv2.LINE_AA)
        
        for track in gloc.contributors.values():
            track_color = COLORS[track.node_id]
            sample = self.to_image_coord(track.world_coord)
            convas = cv2.line(convas, mean, sample, track_color, thickness=1, lineType=cv2.LINE_AA)
            convas = cv2.circle(convas, sample, radius=RADIUS_LOCAL, color=track_color, thickness=-1, lineType=cv2.LINE_AA)
        return convas
        
    def to_image_coord(self, world_coord:Point) -> tuple[float,float]:
        pt_m = self.localizer.from_world_coord(world_coord.xy)
        return tuple(Point(self.localizer.to_image_coord(pt_m)).to_rint().xy)
    
    
def consume_track_events(source:Generator[NodeTrack, None, None],
                      locator:GlobalTrackLocator,
                      drawer: GlobalTrackDrawer,
                      start_index:int):
    current_ts = -1
    
    with closing(drawer) as drawer:
        for track in source:
            if current_ts < 0:
                current_ts = track.ts   
            elif current_ts < track.ts:
                if track.frame_index >= start_index:
                    drawer.draw_tracks(locator.locations)
                    # delay_ms = (track.ts - current_ts)
                    delay_ms = 1
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord('q'):
                        break
                current_ts = track.ts
            else:
                pass
                
            # update global track locations
            locator.update(track)



def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    world_image = cv2.imread("regions/etri_testbed/ETRI_221011.png", cv2.IMREAD_COLOR)
    localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                         camera_index=0, contact_point=ContactPointType.Simulation)
    drawer = GlobalTrackDrawer(title="Multiple Objects Tracking", localizer=localizer, world_image=world_image)
    
    consumer = KafkaConsumer(bootstrap_servers=args.kafka_brokers,
                             auto_offset_reset=args.kafka_offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['global-tracks'])
    
    done = False
    last_ts = -1
    tracks:list[GlobalTrack] = []
    while not done:
        for record in read_topics(consumer, timeout_ms=500):
            track = GlobalTrack.deserialize(record.value)
            
            if last_ts < 0:
                last_ts = track.ts
                
            if last_ts != track.ts:
                drawer.draw_tracks(tracks)
                
                # delay_ms = (track.ts - current_ts)
                delay_ms = track.ts - last_ts
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