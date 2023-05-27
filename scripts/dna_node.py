import os
from contextlib import closing
from datetime import timedelta

import yaml
from omegaconf import OmegaConf

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import dna
from dna import config
from scripts.utils import load_camera_conf
from dna.camera import ImageProcessor, create_opencv_camera_from_conf
from dna.node.node_processor import build_node_processor

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Track objects and publish their locations")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    
    parser.add_argument("--camera", metavar="uri", help="target camera uri")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--begin_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the first frame number")
    parser.add_argument("--end_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the last frame number")

    parser.add_argument("--output", metavar="json file", help="track event file.", default=None)
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default=None)
    parser.add_argument("--loop", action='store_true')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def parse_true_false_string(truth:str):
    truth = truth.lower()
    if truth in ['yes', 'true', 'y', 't', '1']:
        return True
    elif truth in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        return None
    
def update_namespace_with_environ(args:argparse.Namespace) -> argparse.Namespace:
    args = vars(args)
    if v := os.environ.get('DNA_NODE_CONF'):
        args['conf'] = v
    if v := os.environ.get('DNA_NODE_CAMERA'):
        args['camera'] = v
    if v := os.environ.get('DNA_NODE_SYNC'):
        args['sync'] = parse_true_false_string(v)
    if v := os.environ.get('DNA_NODE_BEGIN_FRAME'):
        args['begin_frame'] = v
    if v := os.environ.get('DNA_NODE_END_FRAME'):
        args['end_frame'] = v
    if v := os.environ.get('DNA_NODE_OUTPUT'):
        args['output'] = v
    if v := os.environ.get('DNA_NODE_OUTPUT_VIDEO'):
        args['output_video'] = v
    if v := os.environ.get('DNA_NODE_SHOW_PROGRESS'):
        args['show_progress'] = parse_true_false_string(v)
    if v := os.environ.get('DNA_NODE_SHOW'):
        truth = parse_true_false_string(v)
        if truth is None:
            args['show'] = v
        elif truth is True:
            args['show'] = '0x0'
        else:
            args['show'] = None
    if v := os.environ.get('DNA_NODE_LOGGER'):
        args['logger'] = v
    return argparse.Namespace(**args)
    

def main():
    args, _ = parse_args()
    args = update_namespace_with_environ(args)

    dna.initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # 카메라 설정 정보 추가
    config.update(conf, 'camera', load_camera_conf(args))
    camera = create_opencv_camera_from_conf(conf.camera)

    # args에 포함된 ImageProcess 설정 정보를 추가한다.
    config.update_values(conf, args, 'show', 'output_video', 'show_progress')
    if args.output:
        # 'output'이 설정되어 있으면, track 결과를 frame 단위로 출력할 수 있도록 설정을 수정함.
        OmegaConf.update(conf, "publishing.plugins.output", args.output, merge=True)

    while True:
        options = config.to_dict(config.filter(conf, 'show', 'output_video', 'show_progress'))
        img_proc = ImageProcessor(camera.open(), **options)

        build_node_processor(img_proc, conf)
        
        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
	main()