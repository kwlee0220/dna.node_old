from __future__ import annotations
from typing import Union, List, Tuple, Generator, Iterable, Optional, ByteString

from contextlib import closing

from .types import TrackEvent, TrackId, NodeId, TrackFeature
from .zone import Motion
from .tracklet import Tracklet, TrackletMeta
from dna.assoc import TrajectoryFragment, TrackletId, Association
from dna.support import sql_utils, iterables


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
        zone_relation character varying,
        frame_index integer,
        ts bigint,
        CONSTRAINT track_events_pkey PRIMARY KEY (row_no)
    )
    """
_CREATE_INDEX_ON_TRACK_EVENTS = """
    CREATE INDEX node_track_idx ON track_events(node_id, track_id)
    """

_CREATE_TRACKLETS = """
    CREATE TABLE IF NOT EXISTS tracklets
    (
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        enter_zone character varying,
        exit_zone character varying,
        motion character varying,
        PRIMARY KEY (node_id, track_id)
    );
    """
_CREATE_INDEX_ON_TRACKLET_ENTERS = """
    CREATE INDEX tracklet_enter_idx ON tracklets(node_id, enter_zone ASC NULLS LAST)
"""
_CREATE_INDEX_ON_TRACKLET_EXITS = """
    CREATE INDEX tracklet_exit_idx ON tracklets(node_id, exit_zone ASC NULLS LAST)
"""

_CREATE_TRACKLET_FEATURES = """
    CREATE TABLE IF NOT EXISTS track_features
    (
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        feature bytea NOT NULL,
        ts bigint NOT NULL,
        PRIMARY KEY (node_id, track_id, ts)
    );
"""

_CREATE_TRAJECTORIES = """
    CREATE TABLE IF NOT EXISTS trajectory_fragments
    (
        traj_id bigint NOT NULL,
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        CONSTRAINT trajectory_fragments_pkey PRIMARY KEY (node_id, track_id)
    )
"""
_CREATE_SEQUENCE_TRAJECTORY = """
    CREATE SEQUENCE trajectory_seq
        START 1;
"""

class TrackletStore:
    def __init__(self, connector:sql_utils.SQLConnector) -> None:
        self.connector = connector
        
    def connect(self):
        return self.connector.connect()
        
    def format(self) -> None:
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(_CREATE_TRACK_EVENTS)
            cursor.execute(_CREATE_INDEX_ON_TRACK_EVENTS)
            cursor.execute(_CREATE_TRACKLETS)
            cursor.execute(_CREATE_INDEX_ON_TRACKLET_ENTERS)
            cursor.execute(_CREATE_INDEX_ON_TRACKLET_EXITS)
            cursor.execute(_CREATE_TRACKLET_FEATURES)
            cursor.execute(_CREATE_TRAJECTORIES)
            cursor.execute(_CREATE_SEQUENCE_TRAJECTORY)
            conn.commit()
            
    def drop(self) -> None:
        sql = 'drop table if exists track_events, tracklets, track_features, trajectories;'
        sql2 = 'drop sequence if exists trajectory_seq;'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            cursor.execute(sql2)
            conn.commit()

    ############################################################################################################
    ############################################## Track Events ################################################
    ############################################################################################################  
    def insert_track_events(self, tracks:Iterable[TrackEvent], batch_size:int=30) -> int:
        with closing(self.connector.connect()) as conn:
            return self.insert_track_events_conn(conn, tracks, batch_size)
        
    def insert_track_events_conn(self, conn, tracks:Iterable[TrackEvent], batch_size:int) -> int:
        count = 0
        cursor = conn.cursor()
        for bulk in iterables.buffer_iterable(tracks, count=batch_size):
            values = [track.to_row() for track in bulk]
            cursor.executemany('INSERT INTO track_events VALUES(default, %s,%s,%s, %s,%s,%s, %s, %s,%s)', values)
            count += len(values)
        conn.commit()
        return count
        
    def read_tracklet(self, node_id:NodeId, track_id:TrackId) -> Tracklet:
        sql = 'select * from track_events where node_id=%s and track_id=%s order by row_no'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (node_id, track_id))
            events = [TrackEvent.from_row(row) for row in cursor.fetchall()]
            return Tracklet(track_id, events)
        
    def list_tracklet_range(self, node_id:NodeId, begin_ts:int, end_ts:int) -> List[str]:
        sql = 'select distinct track_id from track_events where node_id=%s and ts >= %s and ts <= %s'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (node_id, begin_ts, end_ts))
            return [row[0] for row in cursor.fetchall()]

    ############################################################################################################
    ############################################# Track Features ###############################################
    ############################################################################################################   
    def insert_track_features(self, features:Iterable[TrackFeature], batch_size:int=30) -> int:
        with closing(self.connector.connect()) as conn:
            return self.insert_track_features_conn(conn, features, batch_size)
        
    def insert_track_features_conn(self, conn, features:Iterable[TrackFeature], batch_size:int) -> int:
        count = 0
        cursor = conn.cursor()
        for bulk in iterables.buffer_iterable(features, count=batch_size):
            values = [feature.to_row() for feature in bulk]
            cursor.executemany('INSERT INTO track_features VALUES(%s,%s,%s,%s)', values)
            count += len(values)
        conn.commit()
        return count
        
    def read_track_features(self, node_id:NodeId, track_id:TrackId, /,
                            begin_ts:Optional[int]=None,
                            end_ts:Optional[int]=None) -> List[TrackFeature]:
        params = [node_id, track_id]
        sql = 'select * from track_features where node_id=%s and track_id=%s'
        if begin_ts is not None:
            sql = sql + ' and frame_index >= %s'
            params.append(begin_ts)
        if end_ts is not None:
            sql = sql + ' and frame_index < %s'
            params.append(end_ts)
            
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            return [TrackFeature.from_row(row) for row in cursor.fetchall()]


    ############################################################################################################
    ############################################## Track Motions ###############################################
    ############################################################################################################ 
        
    def insert_tracklet_meta_conn(self, conn, motions:Iterable[Motion], batch_size:int) -> int:
        count = 0
        cursor = conn.cursor()
        for bulk in iterables.buffer_iterable(motions, count=batch_size):
            rows = [TrackletMeta.from_motion(motion).to_row() for motion in bulk]
            cursor.executemany('INSERT INTO tracklets VALUES(%s,%s,%s,%s,%s)', rows)
            count += len(rows)
        conn.commit()
        return count
    
    def list_tracklet_metas(self, node_id:str, /,
                            enter_zone:Optional[str]=None, exit_zone:Optional[str]=None) -> List[TrackletMeta]:
        enter_zone_pred = f" and enter_zone = '{enter_zone}'" if enter_zone else ''
        exit_zone_pred = f" and exit_zone = '{exit_zone}'" if exit_zone else ''
        sql = f"select * from tracklets where node_id = '{node_id}'{enter_zone_pred}{exit_zone_pred}"

        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [TrackletMeta.from_row(row) for row in cursor.fetchall()]


    ############################################################################################################
    ################################################# Trajectories #############################################
    ############################################################################################################
    def list_fragments_of_trajectory(self, traj_id:int) -> Optional[TrajectoryFragment]:
        sql = f"select traj_id, node_id, track_id from trajectory_fragments where traj_id = {traj_id}"
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            row = cursor.fetchone()
            return TrajectoryFragment.from_row(row) if row else None
    
    def list_fragments_of_tracklet(self, tracklet:TrackletId) -> List[TrajectoryFragment]:
        sql = ( f"select traj_id, node_id, track_id from trajectory_fragments "
                f"where node_id = '{tracklet.node_id}' and track_id = '{tracklet.track_id}'" )
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [TrajectoryFragment.from_row(row) for row in cursor.fetchall()]
        
    def insert_association(self, assoc:Association) -> int:
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            try:
                trj_id = self.get_trajectory_id_cursor(cursor, assoc.tracklet1)
                if trj_id >= 0:
                    frag = TrajectoryFragment(trj_id, assoc.tracklet2)
                    self.insert_trajectory_fragment_cursor(cursor, frag)
                    return trj_id
                
                trj_id = self.get_trajectory_id_cursor(cursor, assoc.tracklet2)
                if trj_id >= 0:
                    frag = TrajectoryFragment(trj_id, assoc.tracklet1)
                    self.insert_trajectory_fragment_cursor(cursor, frag)
                    return trj_id
                
                trj_id = self.allocate_trajectory_id_cursor(cursor)
                frag1 = TrajectoryFragment(trj_id, assoc.tracklet1)
                frag2 = TrajectoryFragment(trj_id, assoc.tracklet2)
                self.insert_trajectory_fragment_cursor(cursor, frag1, frag2)
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
                
    def get_trajectory_id_cursor(self, cursor, tracklet:Tracklet) -> int:
        sql = ( f"select traj_id, node_id, track_id from trajectory_fragments "
                f"where node_id = '{tracklet.node_id}' and track_id = '{tracklet.track_id}' "
                f"limit 1")
        cursor.execute(sql)
        rows = cursor.fetchall()
        if rows:
            return rows[0][0]
        else:
            return -1
            
    def allocate_trajectory_id_cursor(self, cursor) -> int:
        cursor.execute(f"select nextval('trajectory_seq')")
        return cursor.fetchall()[0][0]
        
    def insert_trajectory_fragment_cursor(self, cursor, *frags:TrajectoryFragment) -> None:
        rows = [frag.to_row() for frag in frags]
        cursor.executemany('INSERT INTO trajectory_fragments VALUES(%s,%s,%s)', rows)
            
            
            
            
            
            
            

        
    def stream_tracks(self, sql:str) -> Generator[TrackEvent, None, None]:
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            for row in cursor.fetchall():
                yield TrackEvent.from_row(row)
        
    def read_first_and_last_track(self, node_id:NodeId, track_id:TrackId) -> Tuple[TrackEvent,TrackEvent]:
        sql_first = 'select * from track_events where node_id=%s and track_id=%s order by ts,row_no limit 1'
        sql_last = 'select * from track_events where node_id=%s and track_id=%s order by ts desc,row_no desc limit 1'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_first, (node_id, track_id))
            first = TrackEvent.from_row(cursor.fetchone())
            
            cursor.execute(sql_last, (node_id, track_id))
            last = TrackEvent.from_row(cursor.fetchone())
            return first, last
        
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