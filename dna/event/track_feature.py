from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from dna import ByteString, NodeId, TrackId, TrackletId
from .types import KafkaEvent
from .proto.reid_feature_pb2 import TrackFeatureProto


@dataclass(frozen=True, eq=True, order=False, repr=False)   # slots=True
class TrackFeature(KafkaEvent):
    node_id: NodeId     # node id
    track_id: TrackId   # tracking object id
    frame_index: int
    ts: int = field(hash=False)
    feature: Optional[np.ndarray] = field(default=None)
    zone_relation: Optional[str] = field(default=None)

    def key(self) -> str:
        return self.node_id

    @property
    def tracklet_id(self) -> TrackletId:
        return TrackletId(self.node_id, self.track_id)

    @staticmethod
    def from_row(row:tuple[NodeId,TrackId,ByteString,str,int,int]) -> TrackFeature:
        feature = np.frombuffer(row[2], dtype=np.float32) if row[2] is not None else None
        return TrackFeature(node_id=row[0], track_id=row[1], feature=feature,
                            zone_relation=row[3], frame_index=row[4], ts=row[5])

    def to_row(self) -> tuple[NodeId,TrackId,ByteString,str,int,int]:
        bfeature = self.feature.tobytes() if self.feature is not None else None
        return (self.node_id, self.track_id, bfeature, self.zone_relation, self.frame_index, self.ts)

    def serialize(self) -> bytes:
        return self.to_bytes()

    def deserialize(binary_data:ByteString) -> TrackFeature:
        return TrackFeature.from_bytes(binary_data)

    def to_bytes(self) -> bytes:
        proto = TrackFeatureProto()
        proto.node_id = self.node_id
        proto.track_id = self.track_id
        if self.feature is not None:
            proto.bfeature = self.feature.tobytes()
        proto.zone_relation = self.zone_relation
        proto.frame_index = self.frame_index
        proto.ts = self.ts

        return proto.SerializeToString()

    @staticmethod
    def from_bytes(binary_data:bytes) -> TrackFeature:
        proto = TrackFeatureProto()
        proto.ParseFromString(binary_data)
        
        feature = np.frombuffer(proto.bfeature, dtype=np.float32) if proto.HasField('bfeature') else None
        return TrackFeature(node_id=proto.node_id, track_id=proto.track_id, feature=feature,
                            zone_relation=proto.zone_relation, frame_index=proto.frame_index, ts=proto.ts)

    def __repr__(self) -> str:
        # dt = utc2datetime(self.ts)
        return f'{self.__class__.__name__}[id={self.node_id}[{self.track_id}], zone={self.zone_relation}, ts={self.ts}]'