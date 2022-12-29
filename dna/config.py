from __future__ import annotations

from typing import Union, Optional, Any, Tuple, List, Dict
from pathlib import Path

from argparse import Namespace 
from omegaconf import OmegaConf


class Config:
    def __init__(self, conf: OmegaConf) -> None:
        self.conf: OmegaConf = conf
        
    @staticmethod
    def load(config_path: Union[str,Path]) -> OmegaConf:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        return Config(OmegaConf.load(config_path))
    
    def exists(self, key_path:str) -> bool:
        for name in key_path.split('.'):
            if hasattr(conf, name):
                conf = conf.get(name)
            else:
                return False
        return True
    
    def _get(self, key_path:List[str], def_value: Optional[Any]=None) -> Any:
        ret = def_value
        for name in key_path:
            if hasattr(ret, name):
                ret = ret.get(name)
            else:
                return def_value
        return ret
    
    def get(self, key_path:str, def_value: Optional[Any]=None) -> Union[Config,Any]:
        ret = self._get(key_path.split('.'))
        return Config(ret) if OmegaConf.is_config(ret) else ret
    
    def traverse(self, start_key_path:str=""):
        root = self._get(start_key_path.split('.'), None)
        if not root:
            return
        elif OmegaConf.is_dict(root):
            yield from Config._traverse(start_key_path, root)
        else:
            yield (start_key_path, root)
            
    def filter(self, cond) -> Config:
        return OmegaConf(Config._filter_value(self.conf))
    
    @staticmethod
    def merge(conf1: Config, conf2: Config) -> Config:
        return Config(OmegaConf.merge(conf1.conf, conf2.conf))
 
    @staticmethod
    def _traverse(path: str, dict_conf: OmegaConf):
        for key, value in OmegaConf.to_container(dict_conf).items():
            if OmegaConf.is_dict(value):
                yield from Config._traverse(value)
            else:
                return (path + "." + key, value)
        
    @staticmethod  
    def _filter_value(value: Any, cond) -> Union[OmegaConf, Any]:
        if OmegaConf.is_dict(value):
            filtered = {}
            for key, value in OmegaConf.to_container(value).items():
                if cond(key) and (ret := Config._filter(value, cond)):
                    filtered[key] = ret
            return OmegaConf(filtered)
        else:
            return value
        
                