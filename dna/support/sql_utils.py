from __future__ import annotations
from typing import List

from dataclasses import dataclass
from omegaconf import OmegaConf

from dna import Point, Box
from dna.support import iterables


def from_sql_point(pt_str:str) -> Point:
    nums = pt_str.replace('(', '').replace(')', '').split(',')
    return Point.from_np([float(n) for n in nums])

def to_sql_point(pt:Point) -> str:
    return f'({pt.x},{pt.y})'

def from_sql_box(pt_str:str) -> Box:
    nums = [float(num) for num in pt_str.replace('(', '').replace(')', '').split(',')]
    br, tl = tuple(Point.from_np(xy) for xy in iterables.buffer_iterable(nums, 2))
    return Box.from_points(tl, br)

def to_sql_box(box:Box) -> str:
    return ','.join([to_sql_point(pt) for pt in box.to_points()])


@dataclass(frozen=True)
class SQLConnector:
    host: str
    dbname: str
    user: str
    password: str
    port: int
    
    @staticmethod
    def from_conf(conf:OmegaConf):
        infos = {key[3:]:value for key, value in dict(conf).items() if key.startswith('db_')}
        return SQLConnector(**infos)
    
    def connect(self):
        import psycopg2
        from dataclasses import asdict
        return psycopg2.connect(**asdict(self))