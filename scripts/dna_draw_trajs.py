
from typing import Union
from contextlib import closing
from collections import defaultdict

import cv2
from omegaconf import OmegaConf

from dna import Box, Image, BGR, color, Frame, Point
from dna.camera import create_opencv_camera
from dna.event import NodeTrack, read_event_file
from dna.node.world_coord_localizer import WorldCoordinateLocalizer, ContactPointType
from dna.node.running_stabilizer import RunningStabilizer
from dna.node.trajectory_drawer import TrajectoryDrawer


_contact_point_choices = [t.name.lower() for t in ContactPointType]

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Draw paths")
    parser.add_argument("track_file")
    parser.add_argument("--video", metavar="uri", help="video uri for background image")
    parser.add_argument("--frame", metavar="number", default=1, type=int, help="video frame number")
    parser.add_argument("--node", metavar="id", type=str, default=None, help="node id")
    parser.add_argument("--camera_index", metavar="index", type=int, default=0, help="camera index")
    parser.add_argument("--contact_point", metavar="contact-point type", 
                        choices=_contact_point_choices, type=str.lower, default='simulation',
                        help="contact-point type, default=simulation")
    parser.add_argument("--world_view", action='store_true', help="show trajectories in world coordinates")
    parser.add_argument("--thickness", metavar="number", type=int, default=1, help="drawing line thickness")
    parser.add_argument("--interactive", "-i", action='store_true', help="show trajectories interactively")
    parser.add_argument("--pause", action='store_true', help="pause before termination")
    parser.add_argument("--look_ahead", metavar='count', type=int, default=7, help="look-ahead/behind count")
    parser.add_argument("--smoothing", metavar='value', type=float, default=1, help="stabilization smoothing factor")
    parser.add_argument("--color", metavar='color', default='RED', help="color for trajectory")
    parser.add_argument("--output", "-o", metavar="file path", help="output jpeg file path")
    return parser.parse_known_args()

def load_video_image(video_file:str, frame_no:int) -> Image:
    camera = create_opencv_camera(uri=video_file, begin_frame=frame_no)
    with closing(camera.open()) as cap:
        frame:Frame = cap()
        return frame.image if frame is not None else None

def load_track_events(track_file:str, node_id:str) -> defaultdict[str,list[Box]]:
    import functools
    def append(t_boxes, ev):
        t_boxes[ev.track_id].append(ev.location)
        return t_boxes
    def filter_cond(ev):
        return (not node_id or ev.node_id == node_id) and not ev.is_deleted()
        
    events = read_event_file(track_file, event_type=NodeTrack)
    events = filter(filter_cond, events)
    return functools.reduce(append, events, defaultdict(list))

def to_point_sequence(trajs: dict[str,list[Box]]) -> dict[str,list[Point]]:
    def tbox_to_tpoint(tbox:list[Box]):
        return [box.center() for box in tbox]
    return {luid:tbox_to_tpoint(tbox) for luid, tbox in trajs.items()}

def to_contact_points(trajs: dict[str,list[Box]], localizer:WorldCoordinateLocalizer):
    def tbox_to_tpoint(tbox:list[Box]):
        return [Point(localizer.select_contact_point(box.tlbr)) for box in tbox]
    return {luid:tbox_to_tpoint(tbox) for luid, tbox in trajs.items()}
     

def main():
    args, _ = parse_args()

    box_trajs = load_track_events(args.track_file, args.node)

    bg_img = load_video_image(args.video, args.frame)
    contact_point = ContactPointType(_contact_point_choices.index(args.contact_point))
    localizer = WorldCoordinateLocalizer('regions/etri_testbed/etri_testbed.json', args.camera_index,
                                        contact_point=contact_point) if args.camera_index >= 0 else None
    world_image = cv2.imread("data/ETRI_221011.png", cv2.IMREAD_COLOR)

    stabilizer = None
    if args.look_ahead > 0 and args.smoothing > 0:
        stabilizer = RunningStabilizer(args.look_ahead, args.smoothing)

    traj_color = color.__dict__[args.color]
    drawer = TrajectoryDrawer(box_trajs, camera_image=bg_img, world_image=world_image,
                              localizer=localizer, stabilizer=stabilizer, traj_color=traj_color)

    if args.interactive:
        kw_args = {'capture_file':args.output} if args.output else dict()
        drawer.draw_interactively(**kw_args)
    else:
        drawer.show_stabilized = False
        drawer.show_world_coords = args.world_view
        convas = drawer.draw(pause=args.pause or args.output is None)
        if args.output is not None:
            cv2.imwrite(args.output, convas)


if __name__ == '__main__':
    main()