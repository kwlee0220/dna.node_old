
from cv2 import merge
from omegaconf import OmegaConf
import sys

import dna
from dna import config, initialize_logger
from dna.camera import ImageProcessor, create_opencv_camera_from_conf
from scripts.utils import load_camera_conf


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    
    parser.add_argument("--camera", metavar="uri", default=argparse.SUPPRESS, help="target camera uri")
    parser.add_argument("--begin_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the first frame number")
    parser.add_argument("--end_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the last frame number")
    parser.add_argument("--nosync", action='store_true')

    parser.add_argument("--show", "-s", nargs='?', const='0x0', default='0x0')
    parser.add_argument("--loop", action='store_true')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # 카메라 설정 정보 추가
    config.update(conf, 'camera', load_camera_conf(args))
    camera = create_opencv_camera_from_conf(conf.camera)

    # args에 포함된 ImageProcess 설정 정보를 추가한다.
    config.update_values(conf, args, 'show')

    while True:
        img_proc = ImageProcessor(camera.open(), show=conf.show)
        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
    main()