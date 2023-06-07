from __future__ import annotations

from typing import Any, Dict

from argparse import Namespace
from omegaconf import OmegaConf

from dna import config


def load_camera_conf(args:Dict[str,Any]|Namespace) -> OmegaConf:
    if isinstance(args, Namespace):
        args = vars(args)
    
    conf = OmegaConf.create()
    if (v := args.get("camera", None)):
        conf.uri = v
    if (v := args.get("begin_frame", None)):
        conf.begin_frame = v
    if (v := args.get("end_frame", None)):
        conf.end_frame = v
    if (v := args.get("sync", None)) is not None:
        conf.sync = v
    if (v := args.get("nosync", None)) is not None:
        conf.sync = not v
    if (v := args.get("ffmpeg_path", None)):
        conf.ffmpeg_path = v
        
    return conf


def parse_true_false_string(truth:str):
    truth = truth.lower()
    if truth in ['yes', 'true', 'y', 't', '1']:
        return True
    elif truth in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        return None


def update_namespace_with_environ(args:Namespace) -> Namespace:
    import os

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
    if v := os.environ.get('DNA_NODE_KAFKA_BROKERS'):
        brokers = v.split(',')
        args['bootstrap_servers'] = brokers
    if v := os.environ.get('DNA_NODE_LOGGER'):
        args['logger'] = v
    if v := os.environ.get('DNA_NODE_FFMPEG_PATH'):
        args['ffmpeg_path'] = v
        
    return Namespace(**args)