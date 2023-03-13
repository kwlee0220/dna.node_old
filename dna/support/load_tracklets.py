from __future__ import annotations
from typing import Tuple, List, Dict, Set, Optional, Any, Generator

from collections import defaultdict

from dna.tracker import TrackState
from dna.node import TrackEvent


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

def load_tracklets_by_frame(tracklet_gen:Generator[TrackEvent, None, None]) -> Dict[int,List[TrackEvent]]:
    tracklets:Dict[int,List[TrackEvent]] = dict()
    for track in tracklet_gen:
        tracks = tracklets.get(track.frame_index)
        if tracks is None:
            tracklets[track.frame_index] = [track]
        else:
            tracklets[track.frame_index].append(track)

    return tracklets
