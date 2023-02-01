import warnings
warnings.filterwarnings("ignore")

from contextlib import closing
from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import dna
from dna.conf import load_node_conf, get_config
from dna.camera import Camera, ImageProcessor, create_camera_from_conf
from dna.tracker import TrackingPipeline
from dna.support import Option
from scripts.utils import load_camera_conf


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects from a camera")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    
    parser.add_argument("--camera", metavar="uri", help="target camera uri")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--begin_frame", type=int, metavar="number", help="the first frame number")
    parser.add_argument("--end_frame", type=int, metavar="number", help="the last frame number")

    parser.add_argument("--output", "-o", metavar="csv file", help="output detection file.", default=None)
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default=None)
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    parser.add_argument("--loop", action='store_true')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf(args, ['show', 'show_progress'])
    
    # 카메라 설정 정보 추가
    conf.camera = load_camera_conf(get_config(conf, "camera", OmegaConf.create()), args_conf)
    camera = create_camera_from_conf(conf.camera)

    # ImageProcess 설정 정보 추가
    conf.output = args.output
    conf.output_video = args.output_video
    while True:
        img_proc = ImageProcessor(camera.open(), conf)
        tracker_conf = conf.get('tracker', OmegaConf.create())
        tracker_conf = OmegaConf.merge(tracker_conf, dna.conf.filter(args_conf, ['output']))
        track_pipeline = TrackingPipeline.load(img_proc, tracker_conf)
        img_proc.add_frame_processor(track_pipeline)

        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
	main()