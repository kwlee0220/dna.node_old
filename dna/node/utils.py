from __future__ import annotations
from typing import Union, List, Tuple, Generator, Dict

from omegaconf import OmegaConf

from .types import TrackEvent
from dna.support.text_line_writer import TextLineWriter


def read_node_config(db_conf: OmegaConf, node_id:str) -> OmegaConf:
    from contextlib import closing
    import psycopg2

    with closing(psycopg2.connect(host=db_conf.db_host, dbname=db_conf.db_name,
                                  user=db_conf.db_user, password=db_conf.db_password)) as conn:
        sql = f"select id, camera_conf, tracker_conf, publishing_conf from nodes where id='{node_id}'"
        with closing(conn.cursor()) as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
            if row is not None:
                import json

                conf = OmegaConf.create()
                conf.id = node_id
                if row[1] is not None:
                    conf.camera = OmegaConf.create(json.loads(row[1]))
                if row[2] is not None:
                    conf.tracker = OmegaConf.create(json.loads(row[2]))
                if row[3] is not None:
                    conf.publishing = OmegaConf.create(json.loads(row[3]))

                return conf
            else:
                return None
            

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


class CsvTrackEventWriter(TextLineWriter):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    def handle_event(self, ev:TrackEvent) -> None:
        x1, y1, x2, y2 = tuple(ev.location.tlbr)
        line = f"{ev.frame_index},{ev.track_id},{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f},{ev.state.name},{ev.ts}"
        self.write(line + '\n')
       

class JsonTrackEventGroupWriter(TextLineWriter):
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    def handle_event(self, group:List[TrackEvent]) -> None:
        for track in group:
            self.write(track.to_json() + '\n')

