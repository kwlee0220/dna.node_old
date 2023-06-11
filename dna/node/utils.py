from __future__ import annotations

from collections.abc import Generator

from dna.event import TrackEvent
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

