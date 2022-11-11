from contextlib import closing
from threading import Thread
from datetime import timedelta

import yaml
from omegaconf import OmegaConf

import dna
from dna.camera import ImageProcessor,  create_camera_from_conf
from dna.execution import UserInterruptException
from dna.node.node_processor import build_node_processor
from dna.node.utils import read_node_config


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Track objects in a video file")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--node", metavar="id", help="DNA node id")
    parser.add_argument("--output", metavar="json file", help="track event file.", default=None)
    parser.add_argument("--show", "-s", nargs='?', const='0x0')
    parser.add_argument("--loop", action='store_true')
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
    parser.add_argument("--begin_frame", type=int, metavar="number", help="the first frame number", default=1)
    parser.add_argument("--end_frame", type=int, metavar="number", help="the last frame number")

    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_name", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, _, args_conf = dna.load_node_conf(args, ['show', 'show_progress'])
    
    # 카메라 설정 정보 추가
    conf.camera.begin_frame = args.begin_frame
    conf.camera.end_frame = args.end_frame

    if 'output' in args:
        publishing_conf = conf.get('publishing', OmegaConf.create())
        publishing_conf.output = args.output
        conf.publishing = publishing_conf

    camera = create_camera_from_conf(conf.camera)
    while True:
        img_proc = build_node_processor(camera.open(), conf)
        result: ImageProcessor.Result = img_proc.run()
        if not args.loop or result.failure_cause is not None:
            break
    print(result)

if __name__ == '__main__':
	main()