from datetime import timedelta

from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("conf_path", help="configuration file path")
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()
    conf:OmegaConf = dna.load_config(args.conf_path)

    camera:Camera = dna.camera.create_camera_from_conf(conf.camera)
    proc:ImageProcessor = dna.camera.create_image_processor(camera, OmegaConf.create(vars(args)))

    elapsed, frame_count, fps_measured = proc.run()
    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()