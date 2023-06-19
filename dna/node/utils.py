from __future__ import annotations
from collections import defaultdict

from collections.abc import Callable, Generator
import logging
from typing import Optional, Union

from dna.event import TrackEvent
from dna.event.event_processor import EventProcessor
from dna.event.track_event import TrackEvent
from dna.event.types import TimeElapsed
from dna.support.text_line_writer import TextLineWriter
            

def read_tracks_csv(track_file:str) -> Generator[TrackEvent, None, None]:
    import csv
    with open(track_file) as f:
        reader = csv.reader(f)
        for row in reader:
            yield TrackEvent.from_csv(row)


class JsonTrackEventGroupWriter(TextLineWriter):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    def handle_event(self, group:list[TrackEvent]) -> None:
        for track in group:
            self.write(track.to_json() + '\n')


class GroupByFrameIndex(EventProcessor):
    def __init__(self, min_frame_index_func:Callable[[],int], *, logger:Optional[logging.Logger]=None) -> None:
        EventProcessor.__init__(self)

        self.groups:dict[int,list[TrackEvent]] = defaultdict(list)  # frame index별로 TrackEvent들의 groupp
        self.min_frame_index_func = min_frame_index_func
        self.max_published_index = 0
        self.logger = logger

    def close(self) -> None:
        while self.groups:
            min_frame_index = min(self.groups.keys())
            group = self.groups.pop(min_frame_index, None)
            self._publish_event(group)
        super().close()

    def handle_event(self, ev:Union[TrackEvent,TimeElapsed]) -> None:
        if isinstance(ev, TrackEvent):
            # 만일 새 TrackEvent가 이미 publish된 track event group의 frame index보다 작은 경우
            # late-arrived event 문제가 발생하여 예외를 발생시킨다.
            if ev.frame_index <= self.max_published_index:
                raise ValueError(f'late arrived TrackEvent: {ev}')

            group = self.groups[ev.frame_index]
            group.append(ev)

            # pending된 TrackEvent group 중에서 가장 작은 frame index를 갖는 group을 검색.
            frame_index = min(self.groups.keys())
            group = self.groups[frame_index]
            # frame_index, group = min(self.groups.items(), key=lambda t: t[0])

            # 본 GroupByFrameIndex 이전 EventProcessor들에서 pending된 TrackEvent 들 중에서
            # 가장 작은 frame index를 알아내어, 이 frame index보다 작은 값을 갖는 group의 경우에는
            # 이후 해당 group에 속하는 TrackEvent가 도착하지 않을 것이기 때문에 그 group들을 publish한다.
            min_frame_index = self.min_frame_index_func()
            if not min_frame_index:
                min_frame_index = ev.frame_index

            for idx in range(frame_index, min_frame_index):
                group = self.groups.pop(idx, None)
                if group:
                    if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'publish TrackEvent group: frame_index={idx}, count={len(group)}')
                    self._publish_event(group)
                    self.max_published_index = max(self.max_published_index, idx)

    def __repr__(self) -> str:
        keys = list(self.groups.keys())
        range_str = f'[{keys[0]}-{keys[-1]}]' if keys else '[]'
        return f"{self.__class__.__name__}[max_published={self.max_published_index}, range={range_str}]"


