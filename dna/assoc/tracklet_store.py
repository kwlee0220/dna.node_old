from __future__ import annotations
from typing import List, Tuple, Generator, Iterable, Optional

from contextlib import closing

from dna.event import TrackId, NodeId, TrackletId, TrackEvent, TrackFeature
from ..node.zone import TrackletMotion
from ..node.tracklet import Tracklet
from dna.assoc import TrajectoryFragment, Association
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
        ts bigint NOT NULL,
        CONSTRAINT track_events_pkey PRIMARY KEY (row_no)
    )
    """
_CREATE_INDEX_ON_TRACK_EVENTS = """
    CREATE INDEX track_events_idx ON track_events(node_id, track_id)
    """

_CREATE_TRACKET_ROUTES = """
    CREATE TABLE IF NOT EXISTS tracklet_motions
    (
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        zone_sequence character varying NOT NULL,
        enter_zone character varying,
        exit_zone character varying,
        motion character varying,
        ts bigint NOT NULL,
        PRIMARY KEY (node_id, track_id)
    );
    """
_CREATE_INDEX_ON_TRACKLET_ENTERS = """
    CREATE INDEX tracklet_enter_idx ON tracklet_motions(node_id, enter_zone ASC NULLS LAST)
"""
_CREATE_INDEX_ON_TRACKLET_EXITS = """
    CREATE INDEX tracklet_exit_idx ON tracklet_motions(node_id, exit_zone ASC NULLS LAST)
"""

_CREATE_TRACKLET_FEATURES = """
    CREATE TABLE IF NOT EXISTS track_features
    (
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        feature bytea,
        zone_relation character varying,
        ts bigint NOT NULL,
        PRIMARY KEY (node_id, track_id, ts)
    );
"""

_CREATE_TRAJECTORIES = """
    CREATE TABLE IF NOT EXISTS trajectories
    (
        traj_id bigint NOT NULL,
        node_id character varying NOT NULL,
        track_id character varying NOT NULL,
        PRIMARY KEY (traj_id)
    )
"""
_CREATE_INDEX_ON_TRAJECTORIES = """
    CREATE INDEX trajectories_idx ON trajectories(node_id, track_id)
"""
_CREATE_SEQUENCE_TRAJECTORY = """
    CREATE SEQUENCE trajectory_seq
        START 1;
"""

class TrackletStore:
    def __init__(self, connector:sql_utils.SQLConnector) -> None:
        self.connector = connector
        
    def connect(self):
        '''Tracklet Store를 구성하는 DBMS에 접속한다.'''
        return self.connector.connect()
    
    def close(self) -> None:
        self.connector.close()
        
    def format(self) -> None:
        '''Tracklet Store를 구성하는 DBMS 테이블들을 생성한다.'''
        with closing(self.connect()) as conn:
            cursor = conn.cursor()
            
            cursor.execute(_CREATE_TRACK_EVENTS)                # track_events
            cursor.execute(_CREATE_INDEX_ON_TRACK_EVENTS)       # track_events_idx
            cursor.execute(_CREATE_TRACKET_ROUTES)              # tracklet_motions
            cursor.execute(_CREATE_INDEX_ON_TRACKLET_ENTERS)    # tracklet_enter_idx
            cursor.execute(_CREATE_INDEX_ON_TRACKLET_EXITS)     # tracklet_exit_idx
            cursor.execute(_CREATE_TRACKLET_FEATURES)           # track_features
            cursor.execute(_CREATE_TRAJECTORIES)                # trajectories
            cursor.execute(_CREATE_INDEX_ON_TRAJECTORIES)       # trajectories_idx
            cursor.execute(_CREATE_SEQUENCE_TRAJECTORY)         # trajectory_seq
            conn.commit()
            
    def drop(self) -> None:
        '''Tracklet Store를 구성하는 DBMS 테이블들을 삭제한다.'''
        sql = 'drop table if exists track_events, tracklet_motions, track_features, trajectories;'
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
        with conn.cursor() as cursor:
            for bulk in iterables.buffer_iterable(tracks, count=batch_size):
                values = [track.to_row() for track in bulk]
                cursor.executemany('INSERT INTO track_events VALUES(default, %s,%s,%s, %s,%s,%s, %s, %s,%s)', values)
                count += len(values)
            conn.commit()
        return count
        
    def read_track_events(self, node_id:NodeId, track_id:TrackId) -> Tracklet:
        sql = 'select * from track_events where node_id=%s and track_id=%s order by row_no'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (node_id, track_id))
            events = [TrackEvent.from_row(row) for row in cursor.fetchall()]
            return Tracklet(track_id, events)
        
    def list_tracklets_in_range(self, node_id:NodeId, begin_ts:int, end_ts:int) -> List[TrackletId]:
        """주어진 node에서 주어진 기간동안 TrackEvent를 발생시킨 모든 track들의 식별자를 반환한다.
        TrackEvent 발생이 주어진 시간 구간을 포함하는 모든 track들의 식별자를 반환한다.

        Args:
            node_id (NodeId): 검색 대상 node 식별자.
            begin_ts (int):  검색 시작 시각
            end_ts (int): 검색 종료 시각

        Returns:
            List[str]: TrackEvent 발생이 주어진 시간 구간을 포함하는 모든 track들의 식별자 리스트.
        """
        sql = 'select distinct node_id, track_id from track_events where node_id=%s and ts >= %s and ts <= %s'
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (node_id, begin_ts, end_ts))
            return [TrackletId(row[0], row[1]) for row in cursor.fetchall()]

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
            cursor.executemany('INSERT INTO track_features VALUES(%s,%s,%s,%s,%s)', values)
            count += len(values)
        conn.commit()
        return count
        
    def read_tracklet_features(self, tracklet_id:TrackletId,
                               *,
                               begin_ts:Optional[int]=None,
                               end_ts:Optional[int]=None,
                               skip_eot:bool=True) -> List[TrackFeature]:
        """주어진 track에 의해 생성된 feature 레코드를 검색한다.

        Args:
            node_id (NodeId): 검색 대상 track이 검출되는 노드의 식별자.
            track_id (TrackId): 검색 대상 track의 식별자.
            begin_ts (Optional[int], optional): 검색할 feature의 시작 timestamp. Defaults to None.
            end_ts (Optional[int], optional): 검색할 feature의 종료 timestamp.
                'end_ts'보다 작은 timestamp를 갖는 feature가 검색된다. Defaults to None.
            skip_eot (Optional[bool]): End-of-Track feature 레코드 무시 여부.

        Returns:
            List[TrackFeature]: 검색된 Track record 리스트.
        """
        params = list(tracklet_id)
        sql = 'select * from track_features where node_id=%s and track_id=%s'
        if begin_ts is not None:
            sql = sql + ' and frame_index >= %s'
            params.append(begin_ts)
        if end_ts is not None:
            sql = sql + ' and frame_index < %s'
            params.append(end_ts)
        if skip_eot:
            sql = sql + " and zone_relation <> 'D'"
            
        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [TrackFeature.from_row(row) for row in cursor.fetchall()]


    ############################################################################################################
    ############################################## Track Motions ###############################################
    ############################################################################################################ 
        
    def insert_tracklet_motion_conn(self, conn, motions:Iterable[TrackletMotion], batch_size:int) -> int:
        count = 0
        cursor = conn.cursor()
        for bulk in iterables.buffer_iterable(motions, count=batch_size):
            rows = [motion.to_row() for motion in bulk]
            cursor.executemany('INSERT INTO tracklet_motions VALUES(%s,%s,%s,%s,%s,%s,%s)', rows)
            count += len(rows)
        conn.commit()
        return count
    
    def list_tracklet_motions(self, node_id:str, *,
                              enter_zone:Optional[str]=None, exit_zone:Optional[str]=None) -> List[TrackletMotion]:
        enter_zone_pred = f" and enter_zone = '{enter_zone}'" if enter_zone else ''
        exit_zone_pred = f" and exit_zone = '{exit_zone}'" if exit_zone else ''
        sql = f"select * from tracklet_motions where node_id = '{node_id}'{enter_zone_pred}{exit_zone_pred}"

        with closing(self.connector.connect()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [TrackletMotion.from_row(row) for row in cursor.fetchall()]


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
    
    def list_fragments_of_tracklet(self, node_id:NodeId, track_id:TrackId) -> List[TrajectoryFragment]:
        sql = ( f"select traj_id, node_id, track_id from trajectory_fragments "
                f"where node_id = '{node_id}' and track_id = '{track_id}'" )
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
            cursor.execute("insert into tracklet_motions values (%s,%s,%s,%s,%s,%s,%s) on conflict do update", value)