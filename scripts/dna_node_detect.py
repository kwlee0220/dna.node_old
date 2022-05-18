from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor, create_image_processor
from dna.detect.utils import load_object_detecting_callback


def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("conf_path", help="configuration file path")

    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--output", metavar="file", help="output detection file.", default=None)
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()

    conf = dna.load_config(args.conf_path)

    camera_params = Camera.Parameters.from_conf(conf.camera)
    camera = dna.camera.create_camera(camera_params)

    show = dna.conf.get_config_value(conf.node, 'show').getOrElse(False)
    if show:
        conf.node.window_name = f'id={conf.node.id}, camera={conf.camera.uri}'

    draw_detections = show or args.output is not None
    cb = load_object_detecting_callback(args.detector, args.output, draw_detections)

    proc_params = ImageProcessor.Parameters.from_conf(conf.node)
    proc = create_image_processor(params=proc_params, capture=camera.open(), callback=cb)
    elapsed, frame_count, fps_measured = proc.run()

    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()