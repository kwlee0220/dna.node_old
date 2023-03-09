from contextlib import closing
from datetime import timedelta

import yaml
from omegaconf import OmegaConf

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import dna
from dna.conf import load_node_conf, get_config
from scripts.utils import load_camera_conf
from dna.camera import ImageProcessor,  create_camera_from_conf
from dna.node.node_processor import build_node_processor

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Track objects and publish their locations")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--camera", metavar="uri", help="target camera uri")
    parser.add_argument("--output", metavar="output dir", help="directory for training data", default=None)
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf(args, ['show_progress'])
    
    # 카메라 설정 정보 추가
    conf.camera = load_camera_conf(get_config(conf, "camera", OmegaConf.create()), args_conf)
    camera = create_camera_from_conf(conf.camera)

    if 'output' in args:
        publishing_conf = get_config(conf, 'publishing', OmegaConf.create())
        if args.output:
            publishing_conf.output = args.output 
        conf.publishing = publishing_conf

    img_proc = build_node_processor(camera.open(), conf)
    result: ImageProcessor.Result = img_proc.run()

if __name__ == '__main__':
	main()