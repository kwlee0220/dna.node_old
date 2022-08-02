from contextlib import closing
from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf
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
    dna.initialize_logger()
    
    args, _ = parse_args()
    conf = dna.load_config(args.conf_path)
    dna.conf.update(conf, vars(args), ['show', 'show_progress', 'output_video'])

    if conf.get('show', False) and conf.get('window_name', None) is None:
        conf.window_name = f'camera={conf.camera.uri}'

    camera:Camera = create_camera_from_conf(conf.camera)
    img_proc = ImageProcessor(camera.open(), conf)
    img_proc.callback = load_object_detecting_callback(detector_uri=args.detector, output=args.output,
                                                        draw_detections=img_proc.is_drawing())
    result = img_proc.run()
    print(result)

if __name__ == '__main__':
	main()