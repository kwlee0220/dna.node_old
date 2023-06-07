
from typing import List, Optional

from contextlib import closing

from kafka import KafkaConsumer

from dna import initialize_logger, config
from dna.event import TrackEvent, TrackFeature, TrackletMotion
from dna.support import sql_utils
from dna.event.tracklet_store import TrackletStore


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Tracklet and tracks commands")
    
    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_dbname", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")
    
    parser.add_argument("--bootstrap_servers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--auto_offset_reset", default='earliest', help="A policy for resetting offsets: 'latest', 'earliest', 'none'")
    parser.add_argument('-f', '--format', action='store_true', help='(re-)create tables necessary for Tracklet store.')
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")

    return parser.parse_known_args()


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    store = TrackletStore(sql_utils.SQLConnector.from_conf(conf))
    if conf.format:
        store.drop()
        store.format()
    
    bootstrap_servers = config.get(conf, 'bootstrap_servers', default=['localhost:9092'])
    offset = config.get(conf, 'auto_offset_reset', default='earliest')
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                             auto_offset_reset=offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['track-events', 'track-motions', 'track-features'])
    
    while True:
        partitions = consumer.poll(timeout_ms=500, max_records=100)
        if partitions:
            with closing(store.connect()) as conn:
                for topic_info, partition in partitions.items():
                    if topic_info.topic == 'track-events':
                        tracks = [TrackEvent.deserialize(serialized.value) for serialized in partition]
                        store.insert_track_events_conn(conn, tracks, batch_size=30)
                    elif topic_info.topic == 'track-motions':
                        metas = [TrackletMotion.deserialize(serialized.value) for serialized in partition]
                        store.insert_tracklet_motion_conn(conn, metas, batch_size=30)
                    elif topic_info.topic == 'track-features':
                        features = [TrackFeature.deserialize(serialized.value) for serialized in partition]
                        # features = [feature for feature in features if feature.zone_relation != 'D']
                        store.insert_track_features_conn(conn, features, batch_size=8)
                        

if __name__ == '__main__':
    main()