from typing import List

from omegaconf import OmegaConf
from shapely.geometry import LineString
import cv2

from dna import Point, Image
from dna.color import BGR
from dna.tracker import TrackState
from .local_path_event import LocalPathEvent
from .track_event import TrackEvent
from .event_processor import EventProcessor

class Session:
    def __init__(self, node_id:str, luid:str) -> None:
        self.node_id = node_id
        self.luid = luid

        self.points:List[Point] = []
        self.world_coords:List[Point] = []
        self.first_frame = -1
        self.last_frame = -1

    @property
    def length(self) -> int:
        return len(self.points)

    def append(self, ev: TrackEvent) -> None:
        self.points.append(ev.location.center())
        self.world_coords.append(ev.world_coord)
        if self.first_frame < 0:
            self.first_frame = ev.frame_index
        self.last_frame = ev.frame_index

    def build_local_path(self, length: int, cont: bool) -> LocalPathEvent:
        camera_path = LineString([pt.to_tuple() for pt in self.points[:length]]).wkb_hex
        world_path = LineString([pt.to_tuple() for pt in self.world_coords[:length]]).wkb_hex
        self.points = self.points[length:]
        self.world_coords = self.world_coords[length:]

        return LocalPathEvent(node_id=self.node_id, luid=self.luid,
                              camera_path=camera_path, world_path = world_path,
                              first_frame=self.first_frame, last_frame=self.last_frame,
                              continuation=cont)

    # def draw_camera_path(self, convas:Image, color:BGR, thickness:int=2) -> Image:
    #     return cv2.polylines(convas, [pt.to_tuple() for pt in self.points], False, color, thickness)

    # def draw_world_path(self, convas:Image, color:BGR, thickness:int=2) -> Image:
    #     return cv2.polylines(convas, [pt.to_tuple() for pt in self.world_coords], False, color, thickness)

class LocalPathGenerator(EventProcessor):
    MAX_PATH_LENGTH = 100

    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)

        self.max_path_length = conf.get('max_path_length', LocalPathGenerator.MAX_PATH_LENGTH)
        self.sessions = dict()

    def close(self) -> None:
        super().close()
        
        # build local paths from the unfinished sessions and upload them
        for session in self.sessions.values():
            pev = session.build_local_path(cont=False)
            self.publish_event(pev)
        self.sessions.clear()


    def handle_event(self, ev: TrackEvent) -> None:
        session = self.sessions.get(ev.luid, None)
        if session is None:
            session = Session(ev.node_id, ev.luid)
            self.sessions[ev.luid] = session

        if ev.state == TrackState.Deleted:
            self.sessions.pop(ev.luid, None)
            if session.length > 0:
                pev = session.build_local_path(length=session.length, cont=False)
                self.publish_event(pev)
        else:
            if session.length >= self.max_path_length + 10:
                pev = session.build_local_path(length=self.max_path_length, cont=True)
                self.publish_event(pev)

                # refresh the current session
                # session = Session(ev.node_id, ev.luid)
                # self.sessions[ev.luid] = session
            session.append(ev)