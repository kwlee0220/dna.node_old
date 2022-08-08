from contextlib import closing
from threading import Thread
from datetime import timedelta

import yaml
from omegaconf import OmegaConf

import dna
from dna.camera import ImageProcessor,  create_camera_from_conf
from dna.node.node_processor import build_node_processor
from dna.node.utils import read_node_config


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--conf", metavar="file path", help="configuration file path")
    parser.add_argument("--node", metavar="id", help="DNA node id")
    parser.add_argument("--output", metavar="json file", help="track event file.", default=None)
    parser.add_argument("--show", action='store_true')
    parser.add_argument("--show_progress", help="display progress bar.", action='store_true')
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

    if args_conf.get('node', None) is not None:
        db_conf = dna.conf.filter(args_conf, ['db_host', 'db_port', 'db_name', 'db_user', 'db_password'])
        conf = read_node_config(db_conf, node_id=args_conf.node)
        if conf is None:
            raise ValueError(f"unknown node: id='{args_conf.node}'")
    elif args_conf.get('conf', None) is not None:
        conf = dna.load_config(args_conf.conf)
    else:
        raise ValueError('node configuration is not specified')
    conf = OmegaConf.merge(conf, dna.conf.filter(args_conf, ['show', 'show_progress']))

    if conf.get('show', False) and conf.get('window_name', None) is None:
        conf.window_name = f'camera={conf.camera.uri}'

    if 'output' in args:
        publishing_conf = conf.get('publishing', OmegaConf.create())
        publishing_conf.output = args.output
        conf.publishing = publishing_conf

    camera = create_camera_from_conf(conf.camera)
    img_proc = build_node_processor(camera.open(), conf)
    result = img_proc.run()
    print(result)

if __name__ == '__main__':
	main()