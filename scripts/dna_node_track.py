from contextlib import closing
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import argparse
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor, create_camera_from_conf
from dna.tracker import TrackingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects from a camera")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--node", metavar="id", help="DNA node id")

    parser.add_argument("--output", "-o", metavar="csv file", help="output detection file.", default=None)
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    
    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_name", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")
    return parser.parse_known_args()

def main():
    dna.initialize_logger()
    
    args, _ = parse_args()
    args_conf = OmegaConf.create(vars(args))
    conf:OmegaConf = dna.load_conf_from_args(args_conf)

    if conf.get('show', False) and conf.get('window_name', None) is None:
        conf.window_name = f'camera={conf.camera.uri}'

    camera:Camera = create_camera_from_conf(conf.camera)
    img_proc = ImageProcessor(camera.open(), conf)

    tracker_conf = conf.get('tracker', OmegaConf.create())
    tracker_conf = OmegaConf.merge(tracker_conf, dna.conf.remove_config(args_conf, 'output'))

    track_pipeline = TrackingPipeline.load(img_proc, tracker_conf)
    img_proc.add_frame_processor(track_pipeline)
    
    result = img_proc.run()
    print(result)

if __name__ == '__main__':
	main()