from __future__ import annotations

import os
from typing import Union, Optional, List
from pathlib import Path

from omegaconf import OmegaConf


# class Config:
#     def __init__(self, conf: OmegaConf) -> None:
#         assert conf is not None
#         self.conf = conf

#     @classmethod
#     def load_config_file(cls, config_path: Union[str,Path]) -> None:
#         config_path = Path(config_path) if isinstance(config_path, str) else config_path
#         conf = OmegaConf.load(config_path)
#         return cls(conf)

#     @classmethod
#     def load_sub_config(cls, root_dir:Path, key_path:str) -> Optional[Config]:
#         suffix = key_path.replace('.', '/') + ".yaml"
#         full_path = root_dir / suffix
#         if full_path.is_file():
#             return Config.load_config_file(full_path)
#         else:
#             raise ValueError(f"configuration file is not found: '{full_path}'")

#     def exists(self, key_path:str) -> bool:
#         conf = self.conf
#         parts = key_path.split('.')
#         for name in parts:
#             if hasattr(conf, name):
#                 conf = conf.get(name)
#             else:
#                 return False
#         return True

#     def get(self, key_path:str, def_value: Optional[object]=None) -> object:
#         parts = key_path.split('.')
#         conf = self.conf
#         for name in parts:
#             if hasattr(conf, name):
#                 conf = conf.get(name)
#             else:
#                 return def_value
#         return conf

#     def filter(self, keys: Optional[List[str]]=None) -> None:
#         filtered = { k:self.get(k) for k in keys if self.exists(k) }
#         return Config(OmegaConf.create(filtered))


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

def get_config(conf:OmegaConf, key_path:str, def_value: Optional[object]=None) -> object:
    parts = key_path.split('.')
    for name in parts:
        if hasattr(conf, name):
            conf = conf.get(name)
        else:
            return def_value
    return conf

def filter(conf: OmegaConf, keys: Optional[List[str]]=None) -> None:
    filtered = {k:get_config(conf, k) for k in keys if exists_config(conf, k)}
    return OmegaConf.create(filtered)