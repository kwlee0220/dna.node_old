from contextlib import closing
from datetime import timedelta
from pathlib import Path

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
from dna.support.tracklet_crop_writer import TrackletCropWriter
from dna.support.load_tracklets import read_tracklets_json, load_tracklets_by_frame


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate ReID training data set")
    parser.add_argument("track_video")
    parser.add_argument("track_file")
    parser.add_argument("--type", metavar="[csv|json]", default='csv', help="input track file type")
    parser.add_argument("--margin", metavar="pixels", type=int, default=0, help="margin pixels for cropped image")
    parser.add_argument("--output", "-o", metavar="directory path", help="output training data directory path")
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf(args, ['show_progress'])

    read_tracklets_json(args.track_file)
    tracklets = load_tracklets_by_frame(read_tracklets_json(args.track_file))
    
    # 카메라 설정 정보 추가
    conf.camera = {'uri': args.track_video, 'sync': False, }
    camera = create_camera_from_conf(conf.camera)
    
    img_proc = ImageProcessor(camera.open(), conf)
    training_data_writer = TrackletCropWriter(tracklets, args.output, margin=args.margin)
    img_proc.add_frame_processor(training_data_writer)
    result: ImageProcessor.Result = img_proc.run()

if __name__ == '__main__':
	main()