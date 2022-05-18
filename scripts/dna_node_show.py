from typing import Optional, Dict
from datetime import timedelta
from timeit import default_timer as timer

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("conf_path", help="configuration file path")
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()

    conf = dna.load_config(args.conf_path)

    camera_params = Camera.Parameters.from_conf(conf.camera)
    camera = dna.camera.create_camera(camera_params)

    if conf.node.get('show', False):
        conf.node.window_name = f'id={conf.node.id}, camera={conf.camera.uri}'
    proc_params = ImageProcessor.Parameters.from_conf(conf.node)
    proc = dna.camera.create_image_processor(proc_params, camera.open())
    elapsed, frame_count, fps_measured = proc.run()

    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()