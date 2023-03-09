from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional, Any, Generator
from dataclasses import dataclass, field

from pathlib import Path
import cv2

from dna import Frame, Image, Box
from dna.camera import Camera, ImageProcessor, FrameProcessor
from dna.tracker import TrackState
from dna.node import TrackEvent


@dataclass(eq=True)    # slots=True
class Session:
    track_id: int = field(hash=True)
    dir: Path
    start_frame: int


class TrackletCropWriter(FrameProcessor):
    def __init__(self, tracklets:Dict[int,List[TrackEvent]], output_dir:str, margin:int) -> None:
        self.tracklets = tracklets
        self.margin = margin
        self.sessions:Dict[int,Session] = dict()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def on_started(self, proc:ImageProcessor) -> None:
        pass
    def on_stopped(self) -> None:
        pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        tracks = self.tracklets.pop(frame.index, None)
        if tracks is None:
            return frame
        
        for track in tracks:
            if track.state == TrackState.Deleted:
                self.sessions.pop(track.track_id, None)
            else:
                session = self.sessions.get(track.track_id)
                if session is None:
                    tracklet_dir = self.output_dir / f'{track.track_id}_{track.frame_index}'
                    tracklet_dir.mkdir(parents=True, exist_ok=True)
                    session = Session(track_id=track.track_id, dir=tracklet_dir, start_frame=track.frame_index)
                    self.sessions[track.track_id] = session

                h, w,d = frame.image.shape
                border = Box.from_size((w,h))

                crop_box = track.location.expand(self.margin).to_rint()
                crop_box = crop_box.intersection(border)
                track_crop = crop_box.crop(frame.image)
                crop_file = session.dir / f'{track.frame_index - session.start_frame}.png'
                cv2.imwrite(str(crop_file), track_crop)

        return frame        
