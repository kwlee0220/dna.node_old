from __future__ import annotations

import os
from typing import Union, Optional, List, Tuple
from pathlib import Path

from omegaconf import OmegaConf

DNA_HOME = Path(os.environ.get('DNA_HOME', '.'))
DNA_CONIFIG_FILE = DNA_HOME / 'conf' / 'config.yaml'

DEBUG_FRAME_IDX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = DEBUG_SHOW_IMAGE
DEBUG_START_FRAME = 32
DEBUG_TARGET_TRACKS = None


def load_config(config_path: Union[str,Path], conf_id: Optional[str]=None) -> OmegaConf:
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    conf = OmegaConf.load(config_path)
    return conf[conf_id] if conf_id else conf

def load_sub_config(root_dir:Path, conf_id:str) -> Optional[OmegaConf]:
    suffix = conf_id.replace('.', '/') + ".yaml"
    full_path = root_dir / suffix
    if full_path.is_file():
        return OmegaConf.load(full_path)
    else:
        raise ValueError(f"configuration file is not found: '{full_path}'")

def traverse_configs(root_dir:Path):
    prefix_len = len(Path(os.path.splitext(root_dir)[0]).parts)
    for dir, subdirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".yaml":
                full_path = os.path.join(dir, file)
                parts = Path(os.path.splitext(full_path)[0]).parts[prefix_len:]
                conf_path = '.'.join(parts)
                yield (conf_path, Path(full_path))

from collections import defaultdict
def parse_config_args(args): 
    config_grp = defaultdict(list) 
    for arg in args:
        idx = arg.find('=')
        if idx >= 0:
            key = arg[:idx]
            value = arg[idx+1:]
            idx = key.find('.')
            grp_key = key[:idx]
            key = key[idx+1:]
            config_grp[grp_key].append((key, value))
    return config_grp

def exists_config(conf:OmegaConf, key_path:str) -> bool:
    parts = key_path.split('.')
    for name in parts:
        if hasattr(conf, name):
            conf = conf.get(name)
        else:
            return False
    return True

def get_terminal_config(conf: OmegaConf, key_path: str) -> Optional[Tuple[OmegaConf,str]]:
    parts = key_path.split('.')
    ancester = parts[:-1]
    for key in ancester:
        if hasattr(conf, key):
            conf = conf.get(key)
        else:
            return None
    return (conf, parts[-1])

def get_config(conf:OmegaConf, key_path:str, def_value: Optional[object]=None) -> object:
    parts = key_path.split('.')
    for name in parts:
        if hasattr(conf, name) and conf.get(name):
            conf = conf.get(name)
        else:
            return def_value
    return conf

def filter(conf: OmegaConf, keys: Optional[List[str]]=None) -> OmegaConf:
    filtered = {k:get_config(conf, k) for k in keys if exists_config(conf, k)}
    return OmegaConf.create(filtered)

def remove_config(conf: OmegaConf, key_path:str) -> OmegaConf:
    leaf, key = get_terminal_config(conf, key_path)
    leaf_dict = OmegaConf.to_container(leaf)
    leaf_dict.pop(key, None)
    conf = OmegaConf.create(leaf_dict)
    return conf

def exclude_configs(conf: OmegaConf, keys: Optional[List[str]]=None) -> OmegaConf:
    for key in keys:
        conf = remove_config(conf, key)
    return conf

from argparse import Namespace
from dna.node.utils import read_node_config
_DB_CONF_KEYS = ['db_host', 'db_port', 'db_dbname', 'db_user', 'db_password']
def load_node_conf(args: Namespace,
                   extra_node_configs:List[str]=[]) -> Tuple[OmegaConf,OmegaConf,OmegaConf]: # node_conf, db_conf, args_conf
    args_conf = OmegaConf.create(vars(args)) if isinstance(args, Namespace) else args
    db_conf = filter(args_conf, _DB_CONF_KEYS)
    args_conf = exclude_configs(args_conf, _DB_CONF_KEYS)

    if args_conf.get('node', None) is not None:
        conf = read_node_config(db_conf, node_id=args_conf.node)
        if conf is None:
            raise ValueError(f"unknown node: id='{args_conf.node}'")
    elif args_conf.get('conf'):
        conf = load_config(args_conf.conf)
    else:
        conf = OmegaConf.create()
        
    conf = OmegaConf.merge(conf, filter(args_conf, extra_node_configs))
    args_conf = exclude_configs(args_conf, extra_node_configs)

    return conf, db_conf, args_conf