from __future__ import annotations
from typing import ByteString, Optional, Tuple

import numpy as np

from .types import TrackletId, KafkaEvent
from .proto.reid_feature_pb2 import TrackFeatureProto


class TrackFeature(KafkaEvent):
    __slots__ = ('node_id', 'track_id', '_bfeature', '_feature', 'zone_relation', 'ts')

    def __init__(self, **kwargs) -> None:
        self.node_id:str = kwargs['node_id']
        self.track_id:str = kwargs['track_id']
        self._bfeature:Optional[ByteString] = kwargs.get('bfeature')
        self._feature:Optional[np.ndarray] = kwargs.get('feature')
        self.zone_relation:str = kwargs.get('zone_relation')
        self.ts:int = kwargs['ts']

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    @property
    def feature(self) -> np.ndarray:
        if self._bfeature is None and self._feature is None:
            return None
        if self._feature is None:
            self._feature = np.frombuffer(self._bfeature, dtype=np.float32)
        return self._feature

    @property
    def bfeature(self) -> ByteString:
        if self._bfeature is None and self._feature is None:
            return None
        if not self._bfeature:
            self._bfeature = self._feature.tobytes()
        return self._bfeature

    @staticmethod
    def from_row(args) -> TrackFeature:
        return TrackFeature(node_id=args[0], track_id=args[1], bfeature=args[2], zone_relation=args[3], ts=args[4])

    def to_row(self) -> Tuple[str,str,ByteString,int]:
        return (self.node_id, self.track_id, self.bfeature, self.zone_relation, self.ts)

    def serialize(self) -> bytes:
        return self.to_bytes()

    def deserialize(binary_data:ByteString) -> TrackFeature:
        return TrackFeature.from_bytes(binary_data)

    def to_bytes(self) -> bytes:
        proto = TrackFeatureProto()
        proto.node_id = self.node_id
        proto.track_id = self.track_id
        if self.bfeature is not None:
            proto.bfeature = self.bfeature
        proto.zone_relation = self.zone_relation
        proto.ts = self.ts

        return proto.SerializeToString()

    @staticmethod
    def from_bytes(binary_data:bytes) -> TrackFeature:
        proto = TrackFeatureProto()
        proto.ParseFromString(binary_data)

        bfeature = proto.bfeature if proto.HasField('bfeature') else None
        return TrackFeature(node_id=proto.node_id, track_id=proto.track_id, bfeature=bfeature,
                            zone_relation=proto.zone_relation, ts=proto.ts)

    def __repr__(self) -> str:
        # dt = utc2datetime(self.ts)
        return f'{self.__class__.__name__}[id={self.node_id}[{self.track_id}], zone={self.zone_relation}, ts={self.ts}]'