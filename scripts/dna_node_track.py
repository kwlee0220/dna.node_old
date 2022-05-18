from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import argparse
from omegaconf import OmegaConf

import dna
from dna.conf import get_config_value
from dna.camera import Camera, ImageProcessor, create_image_processor
from dna.track import load_object_tracking_callback


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects from a camera")
    parser.add_argument("conf_path", help="configuration file path")
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()

    conf = dna.load_config(args.conf_path)

    camera_params = Camera.Parameters.from_conf(conf.camera)
    camera = dna.camera.create_camera(camera_params)

    show = get_config_value(conf.node, 'show').getOrElse(False)
    if show:
        conf.node.window_name = f'id={conf.node.id}, camera={conf.camera.uri}'

    domain = dna.Box.from_size(camera.parameters.size)
    output_video = get_config_value(conf.node, 'output_video').getOrNone()
    cb = load_object_tracking_callback(conf.tracker, domain, show, output_video)

    proc_params = ImageProcessor.Parameters.from_conf(conf.node)
    proc = create_image_processor(params=proc_params, capture=camera.open(), callback=cb)
    elapsed, frame_count, fps_measured = proc.run()

    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()