from contextlib import closing
import sys
from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf
from dna.detect.detecting_processor import DetectingProcessor

# __DEFAULT_DETECTOR_URI = 'dna.detect.yolov5:model=l&score=0.4'
__DEFAULT_DETECTOR_URI = 'dna.detect.yolov4'

def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")

    parser.add_argument("--camera", metavar="uri", help="target camera uri")
    parser.add_argument("--begin_frame", type=int, metavar="number", help="the first frame number", default=1)
    parser.add_argument("--end_frame", type=int, metavar="number", help="the last frame number")
    parser.add_argument("--detector", help="Object detection algorithm.", default=None)
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
    conf, _, args_conf = dna.load_node_conf(args, ['show', 'show_progress'])
    
    # 카메라 설정 정보 추가
    conf.camera = dna.conf.get_config(conf, "camera", OmegaConf.create())
    conf.camera.uri = dna.conf.get_config(conf.camera, "uri", args.camera)
    conf.camera.begin_frame = args.begin_frame
    conf.camera.end_frame = args.end_frame

    camera:Camera = create_camera_from_conf(conf.camera)
    
    # detector 설정 정보
    detector_uri = dna.conf.get_config(args_conf, "detector")
    if detector_uri is None:
        detector_uri = dna.conf.get_config(conf, "tracker.dna_deepsort.detector")
    if detector_uri is None:
        print('detector is not specified', file=sys.stderr)

    # ImageProcess 설정 정보 추가
    conf.output = args.output
    conf.output_video = args.output_video
    while True:
        img_proc = ImageProcessor(camera.open(), conf)
        detector = DetectingProcessor.load(detector_uri=detector_uri, output=args.output,
                                            draw_detections=img_proc.is_drawing())
        img_proc.add_frame_processor(detector)
        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
	main()