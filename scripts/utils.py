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
        
    return conf