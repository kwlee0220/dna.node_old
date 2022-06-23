from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor
from dna.detect.utils import load_object_detecting_callback

__DEFAULT_DETECTOR_URI = 'dna.detect.yolov5:model=l&score=0.4'

def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("conf_path", help="configuration file path")

    parser.add_argument("--detector", help="Object detection algorithm.", default=__DEFAULT_DETECTOR_URI)
    parser.add_argument("--output", "-o", metavar="csv file", help="output detection file.", default=None)
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    conf:OmegaConf = dna.load_config(args.conf_path)

    camera:Camera = dna.camera.create_camera(conf.camera)
    proc:ImageProcessor = dna.camera.create_image_processor(camera, OmegaConf.create(vars(args)))

    proc.callback = load_object_detecting_callback(detector_uri=args.detector, output=args.output,
                                                    draw_detections=proc.is_drawing())

    elapsed, frame_count, fps_measured = proc.run()
    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()