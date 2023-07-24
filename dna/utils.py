from __future__ import annotations

from typing import Optional, TypeVar
from collections.abc import Callable
import logging

from datetime import datetime, timezone
from pathlib import Path

from dna import Point
from .color import BGR

T = TypeVar("T")


def datetime2utc(dt: datetime) -> int:
    secs = dt.replace(tzinfo=timezone.utc).timestamp()
    return int(secs * 1000)

def utc2datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000)

def datetime2str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def utc_now_datetime() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_seconds() -> float:
    return datetime.now(timezone.utc).timestamp()

def utc_now_millis() -> int:
    return round(utc_now_seconds() * 1000)


def _parse_keyvalue(kv) -> tuple[str,str]:
    pair = kv.split('=')
    if len(pair) == 2:
        return tuple(pair)
    else:
        return pair, None

def parse_query(query: str) -> dict[str,str]:
    if not query or len(query) == 0:
        return dict()
    return dict([_parse_keyvalue(kv) for kv in query.split('&')])

def get_first_param(args: dict[str,object], key: str, def_value=None) -> object:
    value = args.get(key)
    return value[0] if value else def_value

def split_list(list:list[T], cond) -> tuple[list[T],list[T]]:
    trues = []
    falses = []
    for v in list:
        if cond(v):
            trues.append(v)
        else:
            falses.append(v)
    return trues, falses
    
def remove_cond_from_list(list:list[T], cond:Callable[[T],bool]) -> list[T]:
    length = len(list)
    removeds = []
    for idx in range(length-1, -1, -1):
        if cond(list[idx]):
            removeds.append(list.pop(idx))
    return removeds

def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1

def find_track_index(track_id, tracks):
    return next((idx for idx, track in enumerate(tracks) if track[idx].id == track_id), None)


def gdown_file(url:str, file: Path, force: bool=False):
    if isinstance(file, str):
        file = Path(file)
        
    if force:
        file.unlink()

    if not file.exists():
        # create an empty 'weights' folder if not exists
        file.parent.mkdir(parents=True, exist_ok=True)

        import gdown
        gdown.download(url, str(file.resolve().absolute()), quiet=False)

def initialize_logger(logger_conf_file: Optional[str]=None):
    import yaml
    import logging.config
    
    if logger_conf_file is None:
        import pkgutil
        logger_conf_text = pkgutil.get_data('conf', 'logger.yaml')
    else:
        with open(logger_conf_file, 'rt') as f:
            logger_conf_text = f.read()
    logger_conf = yaml.safe_load(logger_conf_text)
    logging.config.dictConfig(logger_conf)
    
def sub_logger(logger:Optional[logging.Logger], suffix:str) -> Optional[logging.Logger]:
    return logger.getChild(suffix) if logger else None
        
        
def has_method(obj, name:str) -> bool:
    method = getattr(obj, name, None)
    return callable(method) if method else False



from typing import TypeVar, Union
from collections.abc import Callable

T = TypeVar("T")
def get_or_else(value:T, else_value:Union[T,Callable[[],T]]) -> T:
    if value:
        return value
    else:
        return else_value() if callable(else_value) else else_value



def detect_outliers(values:list[T], weight:float=1.5, *,
                    key:Optional[Callable[[T],float]]=None) -> tuple[list[T],list[T]]:
    import numpy as np
    
    keys = [key(v) for v in values] if key else values
    
    v25, v75 = np.percentile(keys, [25, 75])
    iqr = v75 - v25
    step = weight * iqr
    lowest, highest = v25 - step, v75 + step
    
    low_outlier_idxes = [i for i, k in enumerate(keys) if k < lowest]
    high_outlier_idxes = [i for i, k in enumerate(keys) if k > highest]
    return [values[i] for i in low_outlier_idxes], [values[i] for i in high_outlier_idxes]