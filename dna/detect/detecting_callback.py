from __future__ import annotations
from dataclasses import dataclass, field

from typing import Optional, List
from abc import ABCMeta, abstractmethod
from pathlib import Path

from dna import color, BGR, Image, Frame
from dna.camera import ImageProcessor, ImageProcessorCallback
from .object_detector import ObjectDetector, Detection


class DetectingCallback(ImageProcessorCallback):
    __slots__ = 'detector', 'draw_detections', 'box_color', 'label_color', 'show_score', 'output', 'out_handle'
    
    def __init__(self,
                detector:ObjectDetector,
                output: Optional[str]=None,
                draw_detections: bool=False):
        self.detector = detector
        self.draw_detections = draw_detections
        self.box_color = color.RED
        self.label_color = color.WHITE
        self.show_score = True
        self.output = output
        self.out_handle = None

    def on_started(self, proc: ImageProcessor) -> None:
        if self.output:
            Path(self.output).parent.mkdir(exist_ok=True)
            self.out_handle = open(self.output, "w")
        return self

    def on_stopped(self) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        img = frame.image
        frame_idx = frame.index

        for det in self.detector.detect(frame):
            if self.out_handle:
                self.out_handle.write(self._to_string(frame_idx, det) + '\n')
            if self.draw_detections:
                img = det.draw(img, color=self.box_color, label_color=self.label_color, show_score=self.show_score)

        return Frame(image=img, index=frame_idx, ts=frame.ts)

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.label_color = None if self.label_color else color.WHITE
        elif key == ord('c'):
            self.show_score = not self.show_score
        return key

    def _to_string(self, frame_idx: int, det: Detection) -> str:
        tlbr = det.bbox.to_tlbr()
        return (f"{frame_idx},-1,{tlbr[0]:.3f},{tlbr[1]:.3f},{tlbr[2]:.3f},{tlbr[3]:.3f},"
                f"{det.score:.3f},-1,-1,-1,{det.label}")