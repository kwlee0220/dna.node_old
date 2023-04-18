from __future__ import annotations

import os
from typing import Union, Optional, List, Tuple, Dict, Iterable, Any
from pathlib import Path

from argparse import Namespace
from omegaconf import OmegaConf

DNA_HOME = Path(os.environ.get('DNA_HOME', '.'))
DNA_CONIFIG_FILE = DNA_HOME / 'conf' / 'config.yaml'

DEBUG_FRAME_IDX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = DEBUG_SHOW_IMAGE
DEBUG_START_FRAME = 32
DEBUG_TARGET_TRACKS = None


_NOT_EXISTS = OmegaConf.create(None)


def load(config_path:str|Path) -> OmegaConf:
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    return OmegaConf.load(config_path)


def to_conf(value:OmegaConf|Namespace|Dict=dict(), *keys:List[str]) -> OmegaConf:
    if OmegaConf.is_config(value):
        return value
    elif isinstance(value, Namespace):
        return OmegaConf.create(vars(value))
    elif isinstance(value, dict):
        return OmegaConf.create(value)
    else:
        raise ValueError(f'invalid configuration: {value}')


def exists(conf:OmegaConf, key:str) -> bool:
    return OmegaConf.select(conf, key, default=_NOT_EXISTS) != _NOT_EXISTS
    

def get(conf:OmegaConf, key:str, *, default:Optional[Any]=None) -> Any:
    return OmegaConf.select(conf, key, default=default)
    

def get_or_insert_empty(conf:OmegaConf, key:str) -> Any:
    value = OmegaConf.select(conf, key, default=_NOT_EXISTS)
    if value == _NOT_EXISTS:
        value = OmegaConf.create()
        OmegaConf.update(conf, key, value)
    return value


def get_parent(conf:OmegaConf, key:str) -> Tuple[OmegaConf, str, str]:
    last_idx = key.rfind('.')
    if last_idx >= 0:
        parent = OmegaConf.select(conf, key[:last_idx])
        return parent, key[:last_idx], key[last_idx+1]
    else:
        return (None, None, key)


def update(conf:OmegaConf, key:str, value:Dict|Namespace|OmegaConf) -> None:
    values_dict = value
    if isinstance(value, Namespace):
        values_dict = vars(value)
    elif isinstance(value, OmegaConf):
        values_dict = dict(value)

    OmegaConf.update(conf, key, value, merge=True)
    
def update_values(conf:OmegaConf, values:Dict|Namespace|OmegaConf, *keys) -> None:
    values_dict = values
    if isinstance(values, Namespace):
        values_dict = vars(values)
    elif isinstance(values, OmegaConf):
        values_dict = dict(values)
        
    for k, v in values_dict.items():
        if k in keys:
            OmegaConf.update(conf, k, v, merge=True)
    

def filter(conf:OmegaConf, *keys:List[str]) -> OmegaConf:
    return OmegaConf.masked_copy(conf, keys)


def exclude(conf:OmegaConf, *keys:List[str]) -> OmegaConf:
    return OmegaConf.create({k:v for k, v in dict(conf).items() if k not in keys})

        
def to_dict(conf:OmegaConf) -> Dict[str,Any]:
    return OmegaConf.to_container(conf)












# class Configuration:
#     __slots__ = ('conf', )
    
#     def __init__(self, conf:OmegaConf|Namespace|Dict=dict()) -> None:
#         if OmegaConf.is_config(conf):
#             object.__setattr__(self, 'conf', conf)
#         elif isinstance(conf, Namespace):
#             object.__setattr__(self, 'conf', OmegaConf.create(vars(conf)))
#         elif isinstance(conf, dict):
#             object.__setattr__(self, 'conf', OmegaConf.create(conf))
#         else:
#             raise ValueError(f'invalid configuration: {conf}')
    
#     def get(self, key:str, *, default:Optional[Any]=None) -> Any:
#         return OmegaConf.select(self.conf, key, default=default)
    
#     def get_or_empty(self, key:str) -> Any:
#         value = OmegaConf.select(self.conf, key, default=_NOT_EXISTS)
#         if value == _NOT_EXISTS:
#             value = OmegaConf.create()
#         return value
                
#     def __getattr__(self, attrname:str) -> Any:
#         return getattr(self.conf, attrname)
    
#     def __setattr__(self, name: str, value: Any) -> None:
#         if isinstance(value, Configuration):
#             value = value.conf
#         OmegaConf.update(self.conf, name, value, merge=True)

# def load_config(config_path: Union[str,Path], conf_id: Optional[str]=None) -> OmegaConf:
#     config_path = Path(config_path) if isinstance(config_path, str) else config_path
#     conf = OmegaConf.load(config_path)
#     return conf[conf_id] if conf_id else conf

# def load_sub_config(root_dir:Path, conf_id:str) -> Optional[OmegaConf]:
#     suffix = conf_id.replace('.', '/') + ".yaml"
#     full_path = root_dir / suffix
#     if full_path.is_file():
#         return OmegaConf.load(full_path)
#     else:
#         raise ValueError(f"configuration file is not found: '{full_path}'")

# def traverse_configs(root_dir:Path):
#     prefix_len = len(Path(os.path.splitext(root_dir)[0]).parts)
#     for dir, subdirs, files in os.walk(root_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == ".yaml":
#                 full_path = os.path.join(dir, file)
#                 parts = Path(os.path.splitext(full_path)[0]).parts[prefix_len:]
#                 conf_path = '.'.join(parts)
#                 yield (conf_path, Path(full_path))

# def exists_config(conf:OmegaConf, key_path:str) -> bool:
#     parts = key_path.split('.')
#     for name in parts:
#         if hasattr(conf, name):
#             conf = conf.get(name)
#         else:
#             return False
#     return True

# def get_terminal_config(conf: OmegaConf, key_path: str) -> Optional[Tuple[OmegaConf,str]]:
#     parts = key_path.split('.')
#     ancester = parts[:-1]
#     for key in ancester:
#         if hasattr(conf, key):
#             conf = conf.get(key)
#         else:
#             return None
#     return (conf, parts[-1])

# def get_config(conf:OmegaConf, key_path:str, def_value: Optional[object]=None) -> object:
#     parts = key_path.split('.')
#     for name in parts:
#         if hasattr(conf, name) and conf.get(name):
#             conf = conf.get(name)
#         else:
#             return def_value
#     return conf

# def filter(conf:OmegaConf, keys:Iterable[str]=set()) -> OmegaConf:
#     filtered = {k:get_config(conf, k) for k in keys if exists_config(conf, k)}
#     return OmegaConf.create(filtered)

# def remove_config(conf:OmegaConf, key_path:str) -> OmegaConf:
#     leaf, key = get_terminal_config(conf, key_path)
#     leaf_dict = OmegaConf.to_container(leaf)
#     leaf_dict.pop(key, None)
#     conf = OmegaConf.create(leaf_dict)
#     return conf

# def exclude_configs(conf: OmegaConf, keys:Iterable[str]=set()) -> OmegaConf:
#     keys = set(keys)
#     return OmegaConf.create({k:v for k, v in dict(conf).items() if k not in keys})

# from argparse import Namespace
# from dna.node.utils import read_node_config
# _DB_CONF_KEYS = {'db_host', 'db_port', 'db_dbname', 'db_user', 'db_password'}
# def load_node_conf2(args: Namespace,
#                    extra_node_configs:List[str]=[]) \
#     -> Tuple[OmegaConf,OmegaConf,OmegaConf]: # node_conf, db_conf, args_conf
#     args_conf = OmegaConf.create(vars(args)) if isinstance(args, Namespace) else OmegaConf.create(args)
#     db_conf = filter(args_conf, _DB_CONF_KEYS)
#     args_conf = exclude_configs(args_conf, _DB_CONF_KEYS)

#     if args_conf.get('node', None) is not None:
#         conf = read_node_config(db_conf, node_id=args_conf.node)
#         if conf is None:
#             raise ValueError(f"unknown node: id='{args_conf.node}'")
#     elif args_conf.get('conf'):
#         conf = load_config(args_conf.conf)
#     else:
#         conf = OmegaConf.create()
        
#     conf = OmegaConf.merge(conf, filter(args_conf, extra_node_configs))
#     args_conf = exclude_configs(args_conf, extra_node_configs)

#     return conf, db_conf, args_conf

# def load_node_conf(args: Namespace) \
#     -> Tuple[Configuration,Configuration,Configuration]: # node_conf, db_conf, args_conf
#     args_conf = OmegaConf.create(vars(args)) if isinstance(args, Namespace) else OmegaConf.create(args)
#     args_conf = Configuration(args_conf)
#     db_conf = args_conf.filter(_DB_CONF_KEYS)
#     args_conf = args_conf.exclude(_DB_CONF_KEYS)

#     if args_conf.get('node') is not None:
#         conf = read_node_config(db_conf, node_id=args_conf.node)
#         if conf is None:
#             raise ValueError(f"unknown node: id='{args_conf.node}'")
#     elif args_conf.get('conf'):
#         conf = load_config(args_conf.conf)
#     else:
#         conf = OmegaConf.create()
        
#     conf = OmegaConf.merge(conf, filter(args_conf, extra_node_configs))
#     args_conf = exclude_configs(args_conf, extra_node_configs)

#     return conf, db_conf, args_conf