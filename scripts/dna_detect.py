from contextlib import closing
import sys
from datetime import timedelta

import argparse
from omegaconf import OmegaConf

import dna
from dna import config
from dna.camera import ImageProcessor, create_opencv_camera_from_conf
from dna.detect.detecting_processor import DetectingProcessor
from scripts.utils import load_camera_conf

__DEFAULT_DETECTOR_URI = 'dna.detect.yolov5:model=l&score=0.4'
# __DEFAULT_DETECTOR_URI = 'dna.detect.yolov4'

def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    
    parser.add_argument("--camera", metavar="uri", required=False, help="target camera uri")
    parser.add_argument("--sync", action='store_true', help="sync to camera fps")
    parser.add_argument("--begin_frame", type=int, metavar="number", help="the first frame number", default=1)
    parser.add_argument("--end_frame", type=int, metavar="number", default=argparse.SUPPRESS,
                        help="the last frame number")

    parser.add_argument("--detector", help="Object detection algorithm.", default=None)
    parser.add_argument("--output", "-o", metavar="csv file", default=None, help="output detection file.")
    parser.add_argument("--output_video", "-v", metavar="mp4 file", default=argparse.SUPPRESS, help="output video file.")
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default=None)
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    parser.add_argument("--loop", action='store_true')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.load(args.conf) if args.conf else OmegaConf.create()
    
    # 카메라 설정 정보 추가
    config.update(conf, 'camera', load_camera_conf(args))
    camera = create_opencv_camera_from_conf(conf.camera)
    
    # detector 설정 정보
    detector_uri = args.detector
    if detector_uri is None:
        detector_uri = config.get(conf, "tracker.dna_deepsort.detector")
    if detector_uri is None:
        detector_uri = __DEFAULT_DETECTOR_URI
        # print('detector is not specified', file=sys.stderr)

    # args에 포함된 ImageProcess 설정 정보를 추가한다.
    config.update_values(conf, args, 'show', 'output_video', 'show_progress')
    
    while True:
        options = config.to_dict(config.filter(conf, 'show', 'output_video', 'show_progress'))
        img_proc = ImageProcessor(camera.open(), **options)
        
        detector = DetectingProcessor.load(detector_uri=detector_uri,
                                           output=args.output,
                                           draw_detections=img_proc.is_drawing)
        img_proc.add_frame_processor(detector)
        
        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
	main()