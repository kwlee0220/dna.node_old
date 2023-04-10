
from cv2 import merge
from omegaconf import OmegaConf
import sys

import dna
from dna.conf import load_node_conf, get_config
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf
from scripts.utils import load_camera_conf


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    
    parser.add_argument("--camera", metavar="uri", default=argparse.SUPPRESS, help="target camera uri")
    parser.add_argument("--begin_frame", type=int, metavar="number", help="the first frame number", default=1)
    parser.add_argument("--end_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the last frame number")
    parser.add_argument("--nosync", action='store_true')

    parser.add_argument("--show", "-s", nargs='?', const='0x0', default='0x0')
    parser.add_argument("--loop", action='store_true')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = load_node_conf(args, ['show'])
    
    # 카메라 설정 정보 추가
    conf.camera = load_camera_conf(get_config(conf, "camera", OmegaConf.create()), args_conf)
    conf.camera.sync = not args.nosync
    camera = create_camera_from_conf(conf.camera)

    while True:
        img_proc = ImageProcessor(camera.open(), conf)
        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
    main()