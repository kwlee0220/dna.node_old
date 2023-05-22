
from typing import List, Optional

from omegaconf import OmegaConf
from contextlib import closing
import psycopg2
import itertools
from collections import defaultdict

from dna import initialize_logger
from dna import config
from dna.node import TrackEvent
from dna.assoc.tracklet_store import TrackletStore
from dna.node.utils import read_tracks_json
from dna.support import iterables
from dna.support import sql_utils


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Tracklet and tracks commands")
    
    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_dbname", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")

    subparsers = parser.add_subparsers(dest='subparsers')
    
    format_parser = subparsers.add_parser('format')
    format_parser.add_argument('-f', '--force', action='store_true', help='drop them if the tables exist')
    
    format_parser = subparsers.add_parser('drop')

    upload_parser = subparsers.add_parser('upload')
    upload_parser.add_argument("track_files", nargs='+', help="track json files")
    upload_parser.add_argument("--batch", metavar="count", type=int, default=30, help="upload batch count")

    update_parser = subparsers.add_parser('update')
    update_parser.add_argument("node_id", metavar="id", help="target node id")
    update_parser.add_argument("track_id", metavar="id", help="target tracklet id")

    listen_parser = subparsers.add_parser('listen')
    listen_parser.add_argument("--topic", nargs='+', help="Kafka listen port(s)")
    listen_parser.add_argument("--boostrap_servers", default=['localhost:9092'], help="kafka server")
    listen_parser.add_argument("--offset", default='earliest', help="A policy for resetting offsets: 'latest', 'earliest', 'none'")

    return parser.parse_known_args()

def format(conf:OmegaConf, store:TrackletStore) -> None:
    if conf.force:
        store.drop()
    store.format()

def upload(conf:OmegaConf, tracklet_store:TrackletStore) -> None:
    total = 0
    for track_file in conf.track_files:
        count = tracklet_store.insert_track_events(read_tracks_json(track_file))
        print(f'upload track file: {track_file}, count={count}')
        total += count
    print(f'uploaded: total count = {total}')


def update_tracklet(conf:OmegaConf, store:TrackletStore) -> None:
    store.insert_or_update_tracklet(conf.node_id, conf.track_id)
    
    
def listen(store:TrackletStore, bootstrap_servers:List[str], topic:str, *,
           auto_offset_reset:Optional[str]='earliest') -> None:
    from kafka import KafkaConsumer
    from dna.node import TrackFeature
    from dna.node.zone import Motion
    
    # consumer = KafkaConsumer(['track-events', 'track-motions'],
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                             auto_offset_reset=auto_offset_reset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(topic)
    while True:
        partitions = consumer.poll(timeout_ms=500, max_records=100)
        if partitions:
            with closing(store.connect()) as conn:
                for topic_info, partition in partitions.items():
                    match topic_info.topic:
                        case 'track-events':
                            tracks = [TrackEvent.deserialize(serialized.value) for serialized in partition]
                            store.insert_track_events_conn(conn, tracks, batch_size=30)
                        case 'track-motions':
                            metas = [Motion.deserialize(serialized.value) for serialized in partition]
                            store.insert_tracklet_motion_conn(conn, metas, batch_size=30)
                        case 'track-features':
                            features = [TrackFeature.deserialize(serialized.value) for serialized in partition]
                            features = [feature for feature in features if feature.zone_relation != 'D']
                            store.insert_track_features_conn(conn, features, batch_size=8)


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    store = TrackletStore(sql_utils.SQLConnector.from_conf(conf))
    if args.subparsers == 'format':
        format(conf, store)
    elif args.subparsers == 'drop':
        store.drop()
    elif args.subparsers == 'upload':
        upload(conf, store)
    elif args.subparsers == 'listen':
        listen(conf, store)
    
    # total = 0
    # for track_file in args.track_files:
    #     count = store.insert_tracks(read_tracks_json(track_file))
    #     print(f'upload track file: {track_file}, count={count}')
    #     total += count
    # print(f'uploaded: total count = {total}')
    
    # first, last = store.read_first_and_last_track(node_id='etri:04', track_id='1')
    # print(first)
    # print(last)

if __name__ == '__main__':
    main()