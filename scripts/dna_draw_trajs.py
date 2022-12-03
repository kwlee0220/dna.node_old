
from typing import Tuple, List, Dict, Union, Optional
from contextlib import closing
from collections import defaultdict

import numpy as np
import cv2
from omegaconf import OmegaConf

from dna import Box, Image, BGR, color, Frame, Point
from dna.camera import Camera
from dna.camera.utils import create_camera_from_conf
from dna.node.world_coord_localizer import WorldCoordinateLocalizer
from dna.node import stabilizer


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Draw paths")
    parser.add_argument("track_file")
    parser.add_argument("--type", metavar="[csv|json]", default='csv', help="input track file type")
    parser.add_argument("--video", metavar="uri", help="video uri for background image")
    parser.add_argument("--frame", metavar="number", default=1, help="video frame number")
    parser.add_argument("--camera_index", metavar="index", type=int, help="camera index")
    parser.add_argument("--world_view", action='store_true', help="show trajectories in world coordinates")
    parser.add_argument("--thickness", metavar="number", type=int, default=1, help="drawing line thickness")
    parser.add_argument("--interactive", "-i", action='store_true', help="show trajectories interactively")
    parser.add_argument("--stabilize", action='store_true', help="show stabilized coordinates")
    parser.add_argument("--output", "-o", metavar="file path", help="output jpeg file path")
    return parser.parse_known_args()

def load_video_image(video_file:str, frame_no:int) -> Image:
    camera_conf = OmegaConf.create()
    camera_conf.uri = video_file
    camera_conf.begin_frame = frame_no
    camera:Camera = create_camera_from_conf(camera_conf)
    with closing(camera.open()) as cap:
        frame:Frame = cap()
        return frame.image if frame is not None else None

def load_trajectories_csv(track_file:str) -> Dict[str,List[Box]]:
    import csv
    with open(track_file) as f:
        t_boxes = defaultdict(list)
        reader = csv.reader(f)
        for row in reader:
            luid = int(row[1])
            box = Box.from_tlbr([int(v) for v in row[2:6]])
            t_boxes[luid].append(box)
        return t_boxes

def load_trajectories_json(track_file:str) -> Dict[str,List[Box]]:
    import json
    with open(track_file) as f:
        t_boxes = defaultdict(list)
        for line in f.readlines():
            json_obj = json.loads(line)
            luid = int(json_obj['luid'])
            box = Box.from_tlbr(json_obj['location'])
            t_boxes[luid].append(box)
        return t_boxes

def to_point_sequence(trajs: Dict[str,List[Box]]) -> Dict[str,List[Point]]:
    def tbox_to_tpoint(tbox:List[Box]):
        return [box.center() for box in tbox]
    return {luid:tbox_to_tpoint(tbox) for luid, tbox in trajs.items()}

def to_contact_points(trajs: Dict[str,List[Box]], localizer:WorldCoordinateLocalizer):
    def tbox_to_tpoint(tbox:List[Box]):
        return [Point.from_np(localizer.select_contact_point(box.to_tlbr())) for box in tbox]
    return {luid:tbox_to_tpoint(tbox) for luid, tbox in trajs.items()}
            
def _x(pt):
    return pt.x if isinstance(pt, Point) else pt[0]
def _y(pt):
    return pt.y if isinstance(pt, Point) else pt[1]
def _xy(pt):
    return pt.xy if isinstance(pt, Point) else pt

_LOOK_AHEAD = 7
class RunningStabilizer:
    def __init__(self, look_ahead:int) -> None:
        self.look_ahead = look_ahead
        self.current, self.upper = 0, 0
        self.pending_xs: List[float] = []
        self.pending_ys: List[float] = []

    def transform(self, pt:Point) -> Optional[Point]:
        self.pending_xs.append(_x(pt))
        self.pending_ys.append(_y(pt))
        self.upper += 1

        if self.upper - self.current > self.look_ahead:
            xs = stabilizer.stabilization_location(self.pending_xs, self.look_ahead)
            ys = stabilizer.stabilization_location(self.pending_ys, self.look_ahead)
            stabilized = Point(x=xs[self.current], y=ys[self.current])

            self.current += 1
            if self.current > self.look_ahead:
                self.pending_xs.pop(0)
                self.pending_ys.pop(0)
                self.current -= 1
                self.upper -= 1
            return stabilized
        else:
            return None
    
    def get_tail(self) -> List[Point]:
        xs = stabilizer.stabilization_location(self.pending_xs, self.look_ahead)
        ys = stabilizer.stabilization_location(self.pending_ys, self.look_ahead)
        return [Point(x,y) for x, y in zip(xs[self.current:], ys[self.current:])]

    def reset(self) -> None:
        self.current, self.upper = 0, 0
        self.pending_xs: List[float] = []
        self.pending_ys: List[float] = []

class TrajectoryDrawer:
    def __init__(self, box_trajs: Dict[str,List[Box]], bg_image: Image, world_view:bool=False,
                localizer:WorldCoordinateLocalizer=None, world_image: Image=None,
                stabilized:bool=False) -> None:
        self.box_trajs = box_trajs
        self.bg_image = bg_image
        self.world_image = world_image
        self.localizer = localizer
        self.convas = bg_image.copy()
        self.__color = color.RED
        self.__thickness = 2
        self.show_world_coords = world_view
        self.show_contact_point = world_view
        self.stabilizer = RunningStabilizer(_LOOK_AHEAD) if stabilized else None

    @property
    def thickness(self) -> int:
        return self.__thickness

    @thickness.setter
    def thickness(self, value:int) -> None:
        self.__thickness = value

    @property
    def color(self) -> int:
        return self.__color

    @color.setter
    def color(self, value:BGR) -> None:
        self.__color = value

    def draw_to_file(self, outfile:str) -> None:
        convas = self.draw(pause=False)
        cv2.imwrite(outfile, convas)

    def _put_text(self, convas:Image, luid:int=None):
        id_str = f'luid={luid}, ' if luid is not None else ''
        type = "contact" if self.show_contact_point else "centroid"
        view = "world" if self.show_world_coords else "camera"
        stabilized_flag = ', stabilized' if self.stabilizer is not None else ''
        return cv2.putText(convas, f'{id_str}view={view}, {type}{stabilized_flag}',
                            (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, self.__color, 2)

    def draw(self, title='trajectories', pause:bool=True) -> Image:
        bg_image = self.world_image if self.world_image is not None and self.show_world_coords else self.bg_image
        convas = bg_image.copy()
        convas = self._put_text(convas)
        for traj in self.box_trajs.values():
            convas = self._draw_trajectory(convas, traj)
        
        if pause:
            while True:
                cv2.imshow(title, convas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.show_contact_point = not self.show_contact_point
                    convas = bg_image.copy()
                    convas = self._put_text(convas)
                    for traj in self.box_trajs.values():
                        convas = self._draw_trajectory(convas, traj)
                elif key == ord('w') and self.localizer is not None:
                    self.show_world_coords = not self.show_world_coords
                    bg_image = self.world_image if self.world_image is not None and self.show_world_coords else self.bg_image
                    convas = bg_image.copy()
                    convas = self._put_text(convas)
                    for traj in self.box_trajs.values():
                        convas = self._draw_trajectory(convas, traj)
            cv2.destroyWindow(title)
        return convas

    def draw_interactively(self, title='trajectories'):
        id_list = sorted(self.box_trajs)
        try:
            idx = 0
            while True:
                luid = id_list[idx]
                bg_img = self.world_image if self.world_image is not None and self.show_world_coords else self.bg_image
                convas = bg_img.copy()
                convas = self._put_text(convas, luid)
                convas = self._draw_trajectory(convas, self.box_trajs[luid])

                while True:
                    cv2.imshow(title, convas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return
                    elif key == ord('n'):
                        idx = min(idx+1, len(id_list)-1)
                        break
                    elif key == ord('p'):
                        idx = max(idx-1, 0)
                        break
                    elif key == ord('c') and self.localizer is not None:
                        self.show_contact_point = not self.show_contact_point
                        break
                    elif key == ord('w') and self.localizer is not None:
                        self.show_world_coords = not self.show_world_coords
                        break
                    elif key == ord('s'):
                        self.stabilizer = RunningStabilizer(_LOOK_AHEAD) if self.stabilizer is None else None
                        break
        finally:
            cv2.destroyWindow(title)
            
    def _stabilize(self, traj:List[Point]) -> List[Point]:
        pts_s = []
        for pt in traj:
            pt_s = self.stabilizer.transform(pt)
            if pt_s is not None:
                pts_s.append(pt_s)
        pts_s.extend(self.stabilizer.get_tail())
        self.stabilizer.reset()

        return pts_s

    def _draw_trajectory(self, convas:Image, traj: List[Box]) -> Image:
        pts = None
        if self.localizer is not None and self.show_contact_point:
            pts = [self.localizer.select_contact_point(box.to_tlbr()) for box in traj]
        else:
            pts = [box.center() for box in traj]
            
        if self.show_world_coords:
            pts = [self.localizer.to_image_coord(pt) for pt in pts]
            
        if self.stabilizer is not None:
            pts = self._stabilize(pts)
            
        pts = [_xy(pt) for pt in pts]
        pts = np.rint(np.array(pts)).astype('int32')
        return cv2.polylines(convas, [pts], False, self.__color, self.__thickness)

def main():
    args, _ = parse_args()

    box_trajs = load_trajectories_csv(args.track_file) if args.type == 'csv' else load_trajectories_json(args.track_file)

    bg_img = load_video_image(args.video, args.frame)
    localizer = WorldCoordinateLocalizer('conf/region_etri/etri_testbed.json',
                                            args.camera_index) if args.camera_index >= 0 else None
    world_image = cv2.imread("data/ETRI_221011.png", cv2.IMREAD_COLOR)
    drawer = TrajectoryDrawer(box_trajs, bg_img, args.world_view, localizer, world_image)

    if args.output is not None:
        drawer.draw_to_file(args.output)
    elif args.interactive:
        drawer.draw_interactively()
    else:
        drawer.draw(pause=True)

if __name__ == '__main__':
    main()