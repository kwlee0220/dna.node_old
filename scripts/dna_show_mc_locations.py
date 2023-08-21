from __future__ import annotations

from typing import Union, Optional
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import cv2
from omegaconf import OmegaConf
import json

from dna import Box, Image, BGR, color, Frame, Point
from dna.support import iterables
from dna.camera import Camera
from dna.event import NodeTrack, read_json_event_file
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.node import stabilizer
from dna.support import plot_utils
from dna.track.track_state import TrackState

COLORS = [
    color.RED,
    color.BLUE,
    color.GREEN,
    color.YELLOW
]
MAX_DISTS = [65, 55, 45, 43 ]

@dataclass(frozen=True)
class Location:
    track_id: int
    point: Point
    distance: float

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("track_files", nargs='+', help="track files to display")
    parser.add_argument("--offsets", metavar="csv", help="camera frame offsets")
    parser.add_argument("--interactive", "-i", action='store_true', help="show trajectories interactively")
    return parser.parse_known_args()


def load_json(track_file:str, localizer:WorldCoordinateLocalizer) -> tuple[str, dict[int, list[Location]]]:
    def parse_line(line:str) -> tuple[str, str, int, TrackState, Location]:
        ev = NodeTrack.from_json(line)

        if ev.state != TrackState.Deleted:
            pt_m, dist = localizer.from_camera_box(ev.location.tlbr)
            pt = Point(localizer.to_image_coord(pt_m))
            loc = Location(track_id=ev.track_id, point=pt, distance=dist)
        else:
            loc = None
        return (ev.node_id, ev.track_id, ev.frame_index, ev.state, loc)

    with open(track_file) as f:
        node_id = None
        indexed_locations = defaultdict(list)
        state_accum = defaultdict(int)
        for line in f.readlines():
            node_id, track_id, index, state, loc = parse_line(line)
            if state == TrackState.TemporarilyLost:
                state_accum[track_id] += 1
            if state != TrackState.Deleted and state_accum[track_id] < 3:
                indexed_locations[index].append(loc)
            if state == TrackState.Confirmed:
                state_accum[track_id] = 0
            elif state == TrackState.Deleted:
                del state_accum[track_id]
            
    return node_id, dict(indexed_locations)


class MCLocationDrawer:
    def __init__(self, scene_seqs: list[dict[int,Scene]],
                 world_image: Image, offsets:list[int]) -> None:
        self.scene_seqs = scene_seqs
        self.world_image = world_image
        
        frames:set[int] = set()
        for scene_seq in scene_seqs:
            frames.update(scene_seq.keys())
        self.frames = list(frames)
        self.frames.sort()
        self.cursor = 0
        
        self.offsets = offsets

    def draw_frame_index(self, convas: Image, frame_indexes:list[int]) -> Image:
        index_str = ', '.join([str(i) for i in frame_indexes])
        return cv2.putText(convas, f'frames={index_str}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color.RED, 2)
    
    def draw_frame(self, cursor:int) -> Image:
        base_frame = self.frames[cursor]
        frame_indexes = [base_frame + offset for offset in self.offsets] 
        
        convas = self.world_image.copy()
        convas = self.draw_frame_index(convas, frame_indexes)
        for index, frame_index in enumerate(frame_indexes):
            color = COLORS[index]
            scene = self.scene_seqs[index].get(frame_index)
            if scene:
                for vehicle in scene.vehicles:
                    if vehicle.distance < MAX_DISTS[index]:
                        self.draw_vehicle(convas, vehicle, fill_color=color)
        cv2.imshow("locations", convas)
        return convas
    
    def update_offset(self, index:int, delta:int):
        self.offsets[index] += delta
        shift = 0 - min(self.offsets)
        self.offsets = [o + shift for o in self.offsets]
        

    def draw(self, title='locations', interactive:bool=True) -> Image:
        cursor = 0
        self.draw_frame(cursor)
        
        while True:
            delay = 1 if interactive else 100
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif  key == ord(' '):
                interactive = not interactive
            elif interactive:
                if key == ord('n'):
                    if cursor < len(self.frames)-1:
                        cursor += 1
                        self.draw_frame(cursor)
                elif key == ord('p'):
                    if cursor > 0:
                        cursor -= 1
                        self.draw_frame(cursor)
                elif key == ord('s'):
                    image = self.draw_frame(cursor)
                    cv2.imwrite("output/output.png", image)
                else:
                    delta = key - ord('1')
                    if delta >= 0 and delta < 5:
                        self.update_offset(delta, 1)
                        self.draw_frame(cursor)
                    elif delta >= 5 and delta < 10:
                        self.update_offset(delta-5, -1)
                        self.draw_frame(cursor)
            else:
                if key == 0xFF:
                    if cursor < len(self.frames)-1:
                        cursor += 1
                        self.draw_frame(cursor)
        cv2.destroyWindow(title)

    def draw_vehicle(self, convas:Image, vehicle:Vehicle, fill_color:BGR) -> Image:
        center = tuple(vehicle.point.to_rint().xy)
        # convas = cv2.circle(convas, center, radius=3, color=color.RED, thickness=-1, lineType=cv2.LINE_AA)
        convas = plot_utils.draw_label(convas, f'{vehicle.id}', vehicle.point.to_rint(),
                                       color=color.BLACK, fill_color=fill_color, thickness=1)
        return convas


@dataclass(frozen=True)
class Vehicle:
    id: str
    point: Point
    distance: float
    
    @classmethod
    def from_track(cls, track:NodeTrack, localizer:WorldCoordinateLocalizer) -> Vehicle:
        id = f"{track.node_id[5:]}[{track.track_id}]"
        pt_m = localizer.from_world_coord(track.world_coord.xy)
        point = Point(localizer.to_image_coord(pt_m))
        return Vehicle(id=id, point=point, distance=track.distance)
    
@dataclass(frozen=True)
class Scene:
    vehicles: list[Vehicle]

def main():
    args, _ = parse_args()

    if args.offsets is not None:
        offsets = [max(0, int(vstr)) for vstr in args.offsets.split(',')]
    else:
        offsets = [0] * len(args.track_files)
    shift = 0 - min(offsets)
    offsets = [o + shift for o in offsets]
    
    localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json',
                                         camera_index=0, contact_point=ContactPointType.Simulation)
    
    scene_seqs:list[dict[int,Scene]] = list()
    for i, track_file in enumerate(args.track_files):
        tups = [(track.frame_index, Vehicle.from_track(track, localizer))
                    for track in read_json_event_file(track_file, NodeTrack) if not track.is_deleted()]
        scenes:dict[int,list[Vehicle]] = iterables.groupby(tups, key_func=lambda t: t[0], value_func=lambda t: t[1])
        scenario:dict[int,Scene] = {frame_idx:Scene(vehicles=vehicle_list) for frame_idx, vehicle_list in scenes.items()}
        scene_seqs.append(scenario)
        
    world_image = cv2.imread("regions/etri_testbed/ETRI_221011.png", cv2.IMREAD_COLOR)
    drawer = MCLocationDrawer(scene_seqs, world_image, offsets)
    drawer.draw()

if __name__ == '__main__':
    main()