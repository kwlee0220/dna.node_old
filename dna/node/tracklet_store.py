from __future__ import annotations
from typing import Union, List, Tuple, Generator, Iterable, Optional

from contextlib import closing

from dna import Box, Point
from dna.tracker import TrackState
from .types import TrackEvent, TrackId, NodeId
from .tracklet import Tracklet, TrackletMeta
from dna.support import sql_utils, iterables


def from_row(row:Tuple) -> TrackEvent:
    return TrackEvent(node_id=row[1],
                        track_id=row[2],
                        state=TrackState.from_abbr(row[3]),
                        location=sql_utils.from_sql_box(row[4]),
                        world_coord=sql_utils.from_sql_point(row[5]),
                        distance=row[6],
                        frame_index=row[7],
                        ts=row[8])

def from_meta_row(row:Tuple) -> TrackletMeta:
    return TrackletMeta(*row)
    
def to_row(ev:TrackEvent) -> Tuple:
    return (ev.node_id, ev.track_id, ev.state.abbr, sql_utils.to_sql_box(ev.location.to_rint()),
            sql_utils.to_sql_point(ev.world_coord), ev.distance, ev.frame_index, ev.ts)

_CREATE_TRACK_EVENTS = """
    CREATE TABLE IF NOT EXISTS track_events
    (
        row_no bigserial NOT NULL,
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        state character(1) NOT NULL,
        image_location box,
        world_location point,
        distance_to_camera real,
        frame_index integer,
        ts bigint,
        CONSTRAINT track_events_pkey PRIMARY KEY (row_no)
    )
    """
_CREATE_INDEX_ON_TRACK_EVENTS = """
    CREATE INDEX node_track_idx ON track_events(node_id, track_id)
    """

_CREATE_TRACKLETS = """
    CREATE TABLE tracklets
    (
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        enter_zone character varying,
        exit_zone character varying,
        length integer NOT NULL,
        first_ts bigint NOT NULL,
        last_ts bigint NOT NULL,
        PRIMARY KEY (node_id, track_id)
    );
    """
_CREATE_INDEX_ON_TRACKLET_ENTERS = """
    CREATE INDEX tracklet_enter_idx ON tracklets(node_id, enter_zone ASC NULLS LAST)
"""
_CREATE_INDEX_ON_TRACKLET_EXITS = """
    CREATE INDEX tracklet_exit_idx ON tracklets(node_id, exit_zone ASC NULLS LAST)
"""

class TrackletStore:
    def __init__(self, connector:sql_utils.SQLConnector) -> None:
        self.connector = connector
        
    def format(self) -> None:
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(_CREATE_TRACK_EVENTS)
            cursor.execute(_CREATE_INDEX_ON_TRACK_EVENTS)
            cursor.execute(_CREATE_TRACKLETS)
            cursor.execute(_CREATE_INDEX_ON_TRACKLET_ENTERS)
            cursor.execute(_CREATE_INDEX_ON_TRACKLET_EXITS)
            conn.commit()
            
    def drop(self) -> None:
        sql = 'drop table if exists track_events, tracklets;'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()

    def list_tracklets(self, *, node_id:str, track_id:str, first_ts:int, last_ts:int) -> List[TrackletMeta]:
        node_id_term = f"node_id = '{node_id}'" if node_id else None
        track_id_term = f"track_id = '{track_id}'" if track_id else None
        first_ts_term = f"first_ts >= {first_ts}" if first_ts else None
        last_ts_term = f"last_ts >= {last_ts}" if last_ts else None
        where_clause = ' and '.join([term for term in [node_id_term, track_id_term, first_ts_term, last_ts_term] if term])
        sql = 'select * from tracklets where {where_clause}'

        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [from_meta_row(row) for row in cursor.fetchall()]
        
    def stream_tracks(self, sql:str) -> Generator[TrackEvent, None, None]:
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            for row in cursor.fetchall():
                yield from_row(row)
        
    def read_tracklet(self, node_id:NodeId, track_id:TrackId) -> Tracklet:
        sql = 'select * from track_events where node_id=%s and track_id=%s order by row_no'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (node_id, track_id))
            events = [from_row(row) for row in cursor.fetchall()]
            return Tracklet(track_id, events)
        
    def read_first_and_last_track(self, node_id:NodeId, track_id:TrackId) -> Tuple[TrackEvent,TrackEvent]:
        sql_first = 'select * from track_events where node_id=%s and track_id=%s order by ts,row_no limit 1'
        sql_last = 'select * from track_events where node_id=%s and track_id=%s order by ts desc,row_no desc limit 1'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_first, (node_id, track_id))
            first = from_row(cursor.fetchone())
            
            cursor.execute(sql_last, (node_id, track_id))
            last = from_row(cursor.fetchone())
            return first, last
        
    def insert_tracks(self, tracks:Iterable[TrackEvent], batch_size:int=30) -> int:
        with closing(self.connector.connect()) as conn:
            return self._insert_tracks_conn(conn, tracks, batch_size)
        
    def update_or_insert_tracklet(self, node_id:str, track_id:str,
                                  enter_zone:Optional[str]=None, exit_zone:Optional[str]=None) -> None:
        sql_length = 'select count(*) from track_events where node_id=%s and track_id=%s'
        sql_first = 'select ts from track_events where node_id=%s and track_id=%s order by ts, row_no limit 1'
        sql_last = 'select ts from track_events where node_id=%s and track_id=%s order by ts desc, row_no desc limit 1'

        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            
            cursor.execute(sql_length, (node_id, track_id))
            length = cursor.fetchone()[0]

            cursor.execute(sql_first, (node_id, track_id))
            first_ts = cursor.fetchone()[0]
            
            cursor.execute(sql_last, (node_id, track_id))
            last_ts = cursor.fetchone()[0]

            enter_zone = enter_zone if enter_zone else 'NULL'
            exit_zone = exit_zone if exit_zone else 'NULL'
            value = (node_id, track_id, enter_zone, exit_zone, length, first_ts, last_ts)
            cursor.execute("insert into tracklets values (%s,%s,%s,%s,%s,%s,%s) on conflict do update", value)
        
    def _insert_tracks_conn(self, conn, tracks:Iterable[TrackEvent], batch_size:int) -> int:
        cursor = conn.cursor()
        count = 0
        for bulk in iterables.buffer_iterable(tracks, count=batch_size):
            values = [to_row(ev) for ev in bulk]
            cursor.executemany('INSERT INTO track_events VALUES(default, %s,%s,%s,%s,%s, %s,%s,%s)', values)
            count += len(values)
        conn.commit()
        return count