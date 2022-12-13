from typing import Tuple, List
from contextlib import closing, contextmanager, ExitStack

import cv2
import numpy as np

from dna import initialize_logger, Size2d, color, Frame, Image
from dna.camera import create_camera, ImageCapture


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Show multiple videos")
    parser.add_argument("video_uris", nargs='+', help="video uris to display")
    parser.add_argument("--begin_frames", metavar="csv", help="camera offsets")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")

    return parser.parse_known_args()

def draw_frame_index(idx:int, frame: Frame):
    convas = cv2.putText(frame.image, f'{idx}: frames={frame.index}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color.RED, 2)
    return Frame(convas, index=frame.index, ts=frame.ts)

@contextmanager
def multi_camera_context(camera_list):
    with ExitStack() as stack:
        yield [stack.enter_context(closing(camera.open())) for camera in camera_list]

def concatenate(images:List[Image]):
    himg1 = np.concatenate((images[0], images[1]), axis=1)
    himg2 = np.concatenate((images[2], images[3]), axis=1)
    return np.concatenate((himg1, himg2), axis=0)

def synchronize_frames(caps:List[ImageCapture], frame_offsets:List[int]) -> None:
    min_offset = min(frame_offsets)
    frame_offsets = [o - min_offset for o in frame_offsets]

    for camera_idx in range(len(caps)):
        count = frame_offsets[camera_idx]
        if count > 0:
            prev_sync = caps[camera_idx].sync
            caps[camera_idx].sync = False
            for frame_idx in range(count):
                caps[camera_idx]()
            caps[camera_idx].sync = prev_sync

def main():
    args, _ = parse_args()

    initialize_logger(args.logger)

    if args.begin_frames is not None:
        begin_frames = [max(0, int(vstr)) for vstr in args.begin_frames.split(',')]
    else:
        begin_frames = [0] * len(args.video_uris)

    camera_list = [create_camera(uri, begin_frame=begin_frames[idx]) for idx, uri in enumerate(args.video_uris)]
    size:Size2d = (camera_list[0].size() / 2).to_rint()

    blank_image = np.zeros((size.height, size.width, 3), np.uint8)

    camera_list = [camera.resize(size) for camera in camera_list]
    with multi_camera_context(camera_list) as caps:
        while True:
            images = [blank_image]*4
            captured_camera_count = 0
            for idx, cap in enumerate(caps):
                frame:Frame = cap()
                if frame is not None:
                    images[idx] = draw_frame_index(idx, frame).image
                    captured_camera_count += 1
            if captured_camera_count == 0:
                break

            concated = concatenate(images)
            cv2.imshow("multiple cameras", concated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        key = 1
                        break
                    elif key == ord('q'):
                        break
                    elif key - ord('0') < len(caps):
                        camera_idx = key - ord('0')
                        frame = caps[camera_idx]()
                        if frame is not None:
                            images[camera_idx] = draw_frame_index(camera_idx, frame).image
                        else:
                            images[camera_idx] = blank_image
                        concated = concatenate(images)
                        cv2.imshow("multiple cameras", concated)
            if key == ord('q'):
                break
        cv2.destroyWindow("multiple cameras")

if __name__ == '__main__':
    main()