
from typing import Tuple, List, Dict, Union, Optional, NamedTuple
from contextlib import closing
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import cv2
from omegaconf import OmegaConf
import json

from dna import Box, Image, BGR, color, Frame, Point, plot_utils
from dna.camera import Camera
from dna.camera.utils import create_camera_from_conf
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.node import stabilizer, TrackEvent
from dna.tracker import TrackState

COLORS = {
    'etri:04': color.RED,
    'etri:05': color.BLUE,
    'etri:06': color.GREEN,
    'etri:07': color.YELLOW
}
MAX_DISTS = {'etri:04': 50, 'etri:05': 40, 'etri:06': 45, 'etri:07': 40 }

@dataclass(frozen=True)
class Location:
    luid: int
    point: Point
    distance: float

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="show target locations")
    parser.add_argument("track_files", nargs='+', help="track files to display")
    parser.add_argument("--offsets", metavar="csv", help="camera frame offsets")
    parser.add_argument("--interactive", "-i", action='store_true', help="show trajectories interactively")
    return parser.parse_known_args()


def load_json(track_file:str, localizer:WorldCoordinateLocalizer) -> Tuple[str, Dict[int, List[Location]]]:
    def parse_line(line:str) -> Tuple[str, str, int, TrackState, Location]:
        ev = TrackEvent.from_json(line)

        pt_m, dist = localizer.from_camera_box(ev.location.tlbr)
        pt = Point.from_np(localizer.to_image_coord(pt_m))
        loc = Location(luid=ev.track_id, point=pt, distance=dist)
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
    def __init__(self, mc_locations: List[Tuple[str,Dict[int, List[Location]]]],
                 world_image: Image, offsets:List[int]) -> None:
        self.mc_locations = mc_locations
        self.world_image = world_image
        self.indexes = list(mc_locations[0][1].keys())
        self.offsets = offsets

    def draw_frame_index(self, convas: Image, frame_indexes:List[int]) -> Image:
        index_str = ', '.join([str(i) for i in frame_indexes])
        return cv2.putText(convas, f'frames={index_str}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color.RED, 2)
    
    def draw_frame(self, slot_no:int) -> Image:
        index = self.indexes[slot_no]
        frame_indexes = [index + offset for offset in self.offsets]
        
        convas = self.world_image.copy()
        convas = self.draw_frame_index(convas, frame_indexes)
        for index, (node_id, idxed_locations) in zip(frame_indexes, self.mc_locations):
            color = COLORS[node_id]
            for loc in idxed_locations.get(index, []):
                if loc.distance < MAX_DISTS[node_id]:
                    self._draw_circles(convas, loc, fill_color=color)
                    # convas = cv2.circle(convas, loc.point.to_rint().to_tuple(), radius=7, color=color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow("locations", convas)
        return convas

    def draw(self, title='locations', interactive:bool=True) -> Image:
        slot_no = 0
        self.draw_frame(slot_no)
        
        while True:
            delay = 1 if interactive else 100
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif  key == ord(' '):
                interactive = not interactive
            elif interactive:
                if key == ord('n'):
                    if slot_no < len(self.indexes)-1:
                        slot_no += 1
                        self.draw_frame(slot_no)
                elif key == ord('p'):
                    if slot_no > 0:
                        slot_no -= 1
                        self.draw_frame(slot_no)
                elif key == ord('s'):
                    image = self.draw_frame(slot_no)
                    cv2.imwrite("output/output.png", image)
            else:
                if key == 0xFF:
                    if slot_no < len(self.indexes)-1:
                        slot_no += 1
                        self.draw_frame(slot_no)
        cv2.destroyWindow(title)

    def _draw_circles(self, convas:Image, location:Location, fill_color:BGR) -> Image:
        center = location.point.to_rint().to_tuple()
        # convas = cv2.circle(convas, center, radius=3, color=color.RED, thickness=-1, lineType=cv2.LINE_AA)
        convas = plot_utils.draw_label(convas, f'{location.distance:.1f}', location.point.to_rint(),
                                       color=color.BLACK, fill_color=fill_color, thickness=1)
        return convas


def main():
    args, _ = parse_args()

    if args.offsets is not None:
        offsets = [max(0, int(vstr)) for vstr in args.offsets.split(',')]
    else:
        offsets = [0] * len(args.video_uris)

    mc_locations:List[Tuple[str,Dict[int, List[Location]]]] = []
    for i, track_file in enumerate(args.track_files):
        localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', camera_index=i,
                                             contact_point=ContactPointType.Simulation)
        mc_locations.append(load_json(track_file, localizer))
    
    world_image = cv2.imread("data/ETRI_221011.png", cv2.IMREAD_COLOR)

    drawer = MCLocationDrawer(mc_locations, world_image=world_image, offsets=offsets)
    drawer.draw()

if __name__ == '__main__':
    main()