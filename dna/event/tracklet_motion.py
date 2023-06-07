from __future__ import annotations
from typing import Any, ByteString, Tuple
from dataclasses import dataclass, field

import json

from dna.event import KafkaEvent, TrackletId


@dataclass(frozen=True, eq=True)
class TrackletMotion(KafkaEvent):
    node_id: str
    track_id: str
    zone_sequence: str = field(compare=False)
    enter_zone: str = field(compare=False)
    exit_zone: str = field(compare=False)
    motion: str = field(compare=False)
    ts: int

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    @staticmethod
    def from_json(json_str:str) -> TrackletMotion:
        json_obj = json.loads(json_str)
        return TrackletMotion(node_id=json_obj['node'],
                              track_id=json_obj['track_id'],
                              zone_sequence=json_obj['zone_sequence'],
                              enter_zone=json_obj['enter_zone'],
                              exit_zone=json_obj['exit_zone'],
                              motion=json_obj['motion'],
                              ts=json_obj['ts'])

    def to_json(self) -> str:
        serialized = {'node':self.node_id,
                      'track_id':self.track_id,
                      'zone_sequence':self.zone_sequence,
                      'enter_zone':self.enter_zone,
                      'exit_zone':self.exit_zone,
                      'motion':self.motion,
                      'ts':self.ts}
        return json.dumps(serialized, separators=(',', ':'))

    def serialize(self) -> Any:
        return self.to_json().encode('utf-8')

    @staticmethod
    def deserialize(serialized:ByteString) -> TrackletMotion:
        json_str = serialized.decode('utf-8')
        return TrackletMotion.from_json(json_str)

    @staticmethod
    def from_row(row:Tuple) -> TrackletMotion:
        return TrackletMotion(node_id=row[0],
                              track_id=row[1],
                              zone_sequence=row[2],
                              enter_zone=row[3],
                              exit_zone=row[4],
                              motion=row[5],
                              ts=row[6])

    def to_row(self) -> Tuple:
        return (self.node_id, self.track_id, self.zone_sequence,
                self.enter_zone, self.exit_zone, self.motion, self.ts)

    def __repr__(self) -> str:
        # dt = utc2datetime(self.ts)
        return f'{self.__class__.__name__}[id={self.node_id}[{self.track_id}], motion={self.motion}, ts={self.ts}]'