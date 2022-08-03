from contextlib import closing
from threading import Thread
from datetime import timedelta

import yaml
from omegaconf import OmegaConf

import dna
from dna.camera import ImageProcessor,  create_camera_from_conf
from dna.node.node_processor import build_node_processor


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("conf_path", help="configuration file path")
    parser.add_argument("--output", "-o", metavar="json file", help="track event file.", default=None)
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    return parser.parse_known_args()

def main():
    dna.initialize_logger()
    
    args, _ = parse_args()
    conf = dna.load_config(args.conf_path)
    dna.conf.update(conf, vars(args), ['show', 'show_progress'])

    if conf.get('show', False) and conf.get('window_name', None) is None:
        conf.window_name = f'camera={conf.camera.uri}'

    if 'output' in args:
        publishing_conf = conf.get('publishing', OmegaConf.create())
        publishing_conf.output = args.output
        conf.publishing = publishing_conf

    camera = create_camera_from_conf(conf.camera)
    img_proc = build_node_processor(camera.open(), conf)
    result = img_proc.run()
    print(result)

if __name__ == '__main__':
	main()