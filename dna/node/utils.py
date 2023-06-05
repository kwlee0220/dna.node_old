from __future__ import annotations
from typing import List, Generator

from dna.event import TrackEvent
from dna.support.text_line_writer import TextLineWriter
            

def read_tracks_csv(track_file:str) -> Generator[TrackEvent, None, None]:
    import csv
    with open(track_file) as f:
        reader = csv.reader(f)
        for row in reader:
            yield TrackEvent.from_csv(row)


def read_tracks_json(track_file:str) -> Generator[TrackEvent, None, None]:
    import json
    with open(track_file) as f:
        for line in f.readlines():
            yield TrackEvent.from_json(line)
       

class JsonTrackEventGroupWriter(TextLineWriter):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    def handle_event(self, group:List[TrackEvent]) -> None:
        for track in group:
            self.write(track.to_json() + '\n')

