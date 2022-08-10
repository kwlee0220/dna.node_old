
from omegaconf import OmegaConf

from dna.tracker import TrackProcessor


_DEFAULT_MIN_PATH_LENGTH=10

def load_publishing_pipeline(node_id: str, publishing_conf: OmegaConf) -> TrackProcessor:
    from .track_event import TrackEventSource
    from .kafka_event_publisher import KafkaEventPublisher
    from .event_processor import PrintTrackEvent

    source = TrackEventSource(node_id)
    queue = source
    
    # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
    from .refine_track_event import RefineTrackEvent
    refine = RefineTrackEvent()
    queue.add_listener(refine)
    queue = refine

    # drop too-short tracks of an object
    min_path_length = publishing_conf.get('min_path_length', _DEFAULT_MIN_PATH_LENGTH)
    if min_path_length > 0:
        from .drop_short_trail import DropShortTrail
        drop_short_path = DropShortTrail(min_path_length)
        queue.add_listener(drop_short_path)
        queue = drop_short_path

    # attach world-coordinates to each track
    if publishing_conf.get('attach_world_coordinates') is not None:
        from .world_transform import WorldTransform
        world_coords = WorldTransform(publishing_conf.attach_world_coordinates)
        queue.add_listener(world_coords)
        queue = world_coords

    if publishing_conf.get('kafka', None) is not None:
        queue.add_listener(KafkaEventPublisher(publishing_conf.kafka))
    else:
        import logging
        logger = logging.getLogger('dna.node.kafka')
        logger.warning(f'Kafka publishing is not specified')

    if publishing_conf.get('output', None) is not None:
        queue.add_listener(PrintTrackEvent(publishing_conf.output))
        
    return source


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