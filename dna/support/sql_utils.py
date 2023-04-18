from __future__ import annotations
from typing import List

from dataclasses import dataclass, field
from omegaconf import OmegaConf

from dna import Point, Box
from dna.support import iterables


def from_sql_point(pt_str:str) -> Point:
    nums = pt_str.replace('(', '').replace(')', '').split(',')
    return Point([float(n) for n in nums])

def to_sql_point(pt:Point) -> str:
    return f'({pt.x},{pt.y})'

def from_sql_box(pt_str:str) -> Box:
    nums = [float(num) for num in pt_str.replace('(', '').replace(')', '').split(',')]
    br, tl = tuple(Point(xy) for xy in iterables.buffer_iterable(nums, 2))
    return Box.from_points(tl, br)

def to_sql_box(box:Box) -> str:
    pt1 = to_sql_point(Point(box.tl))
    pt2 = to_sql_point(Point(box.br))
    return ','.join([pt1, pt2])


_DB_CONF_KEYS = {'db_host', 'db_port', 'db_dbname', 'db_user', 'db_password'}
@dataclass(frozen=True)
class SQLConnector:
    host: str = field(default='localhost')
    dbname: str = field(default='dna')
    user: str = field(default='dna')
    password: str = field(default='urc2004')
    port: int = field(default=5432)
    
    @staticmethod
    def from_conf(conf:OmegaConf):
        infos = {key[3:]:value for key, value in dict(conf).items() if key in _DB_CONF_KEYS}
        return SQLConnector(**infos)
    
    def connect(self):
        import psycopg2
        from dataclasses import asdict
        return psycopg2.connect(**asdict(self))