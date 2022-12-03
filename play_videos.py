
from typing import Tuple, List, Dict
from contextlib import closing, suppress

from pathlib import Path
import numpy as np
import cv2
from omegaconf import OmegaConf

from dna import Box, Image, BGR, color, Frame, Point
from dna.camera import Camera
from dna.video_writer import VideoWriter
from dna.camera.utils import create_camera_from_conf
from dna.node.world_coord_localizer import WorldCoordinateLocalizer


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Draw paths")
    parser.add_argument("video_file")
    parser.add_argument("output_file")
    parser.add_argument("--begin_frame", type=int, metavar="number", help="the first frame number", default=1)
    parser.add_argument("--end_frame", type=int, metavar="number", help="the last frame number")
    parser.add_argument("--tlbr", metavar="t,l,b,r", help="ROI")
    parser.add_argument("--tlwh", metavar="t,l,w,h", help="ROI")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    conf = OmegaConf.create()
    conf.uri = args.video_file
    conf.begin_frame = args.begin_frame
    conf.end_frame = args.end_frame
    camera:Camera = create_camera_from_conf(conf)
    
    roi:Box = None
    if args.tlbr is not None:
        roi = Box.from_tlbr(np.array([int(s) for s in args.roi.split(',')]))
    elif args.tlwh is not None:
        roi = Box.from_tlwh(np.array([int(s) for s in args.roi.split(',')]))

    with closing(camera.open()) as cap, \
            VideoWriter(args.output_file, cap.fps, roi.size().to_tuple()) as writer:
        while True:
            frame:Frame = cap()
            if frame is None:
                break
            img = roi.project(frame.image) if roi is not None else None
            writer.write(img)

if __name__ == '__main__':
    main()