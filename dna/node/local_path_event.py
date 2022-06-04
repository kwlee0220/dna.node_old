from dataclasses import dataclass, field
import json

from .kafka_event import KafkaEvent


@dataclass(frozen=True, eq=True, order=True)    # slots=True
class LocalPathEvent(KafkaEvent):
    node_id: str
    luid: int
    camera_path:str = field(compare=False, repr=False, hash=False)
    world_path:str = field(compare=False, repr=False, hash=False)
    first_frame: int
    last_frame: int
    continuation:bool = field(compare=False)

    def key(self) -> str:
        return self.node_id.encode('utf-8')
    
    def serialize(self) -> str:
        serialized = { 'node_id': self.node_id, 'luid': self.luid,
                        'camera_path': self.camera_path, 'world_path': self.world_path,
                        'first_frame': self.first_frame, 'last_frame': self.last_frame,
                        'continuation': self.continuation }
        return json.dumps(serialized, separators=(',', ':')).encode('utf-8')