from __future__ import annotations

from typing import Any
from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
import cv2

from dna import Box, Image, plot_utils
from dna.color import BGR


class TrackState(Enum):
    Null = (0, 'N')
    Tentative = (1, 'T')
    Confirmed = (2, 'C')
    TemporarilyLost = (3, 'L')
    Deleted = (4, 'D')
    
    def __init__(self, code, abbr) -> None:
        super().__init__()
        self.code = code
        self.abbr = abbr
    
class DNATrack(metaclass=ABCMeta):
    @property
    @abstractmethod
    def id(self) -> Any: pass

    @property
    @abstractmethod
    def state(self) -> TrackState: pass

    @property
    @abstractmethod
    def location(self) -> Box : pass

    @property
    @abstractmethod
    def frame_index(self) -> int : pass

    @property
    @abstractmethod
    def timestamp(self) -> float : pass

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
    
    def __repr__(self) -> str:
        epoch = int(round(self.timestamp * 1000))
        return f"{self.state.name}[{self.id}]={self.location}, frame={self.frame_index}, ts={epoch}"

    def draw(self, convas:Image, color:BGR, label_color:BGR=None, line_thickness:int=2) -> Image:
        loc = self.location
        convas = loc.draw(convas, color, line_thickness=line_thickness)
        convas = cv2.circle(convas, loc.center().xy.astype(int), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            label = f"{self.id}({self.state.abbr})"
            convas = plot_utils.draw_label(convas, label, loc.tl.astype(int), label_color, color, 2)
        return convas

    def to_string(self) -> str:
        tlbr = self.location.tlbr
        epoch = int(round(self.timestamp * 1000))
        return (f"{self.frame_index},{self.id},{tlbr[0]:.0f},{tlbr[1]:.0f},{tlbr[2]:.0f},{tlbr[3]:.0f},"
                f"{self.state.name},{epoch}")
    
    @staticmethod
    def from_string(csv) -> DNATrack:
        parts = csv.split(',')

        frame_idx = int(parts[0])
        track_id = int(parts[1])
        tlbr = np.array(parts[2:6]).astype('int32')
        bbox = Box.from_tlbr(tlbr)
        state = TrackState(int(parts[6]))
        ts = int(parts[7]) / 1000
        
        return SimpleTrack(id=track_id, state=state, location=bbox, frame_index=frame_idx, timestamp=ts)
    

class SimpleTrack(DNATrack):
    def __init__(self, id:int, state:TrackState, location:Box, frame_index:int, timestamp:float) -> None:
        super().__init__()
        self.__id = id
        self.__state = state
        self.__location = location
        self.__frame_index = frame_index
        self.__timestamp = timestamp
        
    @property
    def id(self) -> Any:
        return self.__id

    @property
    def state(self) -> TrackState:
        return self.__state

    @property
    def location(self) -> Box:
        return self.__location

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    @property
    def timestamp(self) -> float:
        return self.__timestamp