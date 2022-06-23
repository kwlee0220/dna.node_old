from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor, create_image_processor
from dna.tracker.utils import load_object_tracking_callback


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects from a camera")
    parser.add_argument("conf_path", help="configuration file path")
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
    
    tracker_conf = conf.get('tracker', OmegaConf.create())
    tracker_conf.output = args.output
    proc.callback = load_object_tracking_callback(camera, proc, tracker_conf)

    elapsed, frame_count, fps_measured = proc.run()
    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()