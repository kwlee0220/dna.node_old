from __future__ import annotations

from argparse import Namespace
from omegaconf import OmegaConf

from dna import config


def load_camera_conf(args:dict[str,object]|Namespace) -> OmegaConf:
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
    import logging
    from typing import Optional
    from collections.abc import Callable
    
    def set_from_environ(args:Namespace, env_name:str, key:str, *, handler:Optional[Callable[[str],object]]=None) -> None:
        if value := os.environ.get(env_name):
            if handler:
                value = handler(value)
            args[key] = value
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"use environment: name='{env_name}', value='{value}'")
    
    logger = logging.getLogger('dna.envs')

    args = vars(args)
    set_from_environ(args, 'DNA_NODE_CONF', 'conf')
    set_from_environ(args, 'DNA_NODE_CAMERA', 'camera')
    set_from_environ(args, 'DNA_NODE_SYNC', 'sync', handler=parse_true_false_string)
    set_from_environ(args, 'DNA_NODE_BEGIN_FRAME', 'begin_frame', handler=lambda s:int(s))
    set_from_environ(args, 'DNA_NODE_END_FRAME', 'end_frame', handler=lambda s:int(s))
    set_from_environ(args, 'DNA_NODE_OUTPUT', 'output')
    set_from_environ(args, 'DNA_NODE_OUTPUT_VIDEO', 'output_video')
    set_from_environ(args, 'DNA_NODE_SHOW_PROGRESS', 'show_progress', handler=parse_true_false_string)
    
    def parse_size(size_str:str) -> Optional[str]:
        truth = parse_true_false_string(size_str)
        if truth is None:
            return truth
        elif truth is True:
            return '0x0'
        else:
            return None
    set_from_environ(args, 'DNA_NODE_SHOW', 'show', handler=parse_size)
            
    set_from_environ(args, 'DNA_NODE_KAFKA_BROKERS', 'kafka_brokers', handler=lambda s:s.split(','))
    set_from_environ(args, 'DNA_NODE_LOGGER', 'logger')
    set_from_environ(args, 'DNA_NODE_CONF_ROOT', 'conf_root')
    set_from_environ(args, 'DNA_NODE_FFMPEG_PATH', 'ffmpeg_path')
    set_from_environ(args, 'DNA_NODE_RABBITMQ_URL', 'rabbitmq_url')
        
    return Namespace(**args)