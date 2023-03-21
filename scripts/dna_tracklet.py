
from typing import Tuple, List, Dict, Union, Generator

from omegaconf import OmegaConf
from contextlib import closing
import psycopg2
import itertools

from dna import initialize_logger
from dna.conf import load_node_conf
from dna.node import TrackEvent
from dna.node.tracklet_store import TrackletStore
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

    upload_parser = subparsers.add_parser('upload')
    upload_parser.add_argument("track_files", nargs='+', help="track json files")
    upload_parser.add_argument("--batch", metavar="count", type=int, default=30, help="upload batch count")

    update_parser = subparsers.add_parser('update')
    update_parser.add_argument("node_id", metavar="id", help="target node id")
    update_parser.add_argument("track_id", metavar="id", help="target tracklet id")


    # parser.add_argument("track_files", nargs='+', help="track json files")
    # parser.add_argument("--batch", metavar="count", type=int, default=30, help="upload batch count")
    return parser.parse_known_args()

def format(args_conf, store:TrackletStore) -> None:
    if args_conf.force:
        store.drop()
    store.format()

def upload(args_conf, tracklet_store:TrackletStore) -> None:
    total = 0
    for track_file in args_conf.track_files:
        count = tracklet_store.insert_tracks(read_tracks_json(track_file))
        print(f'upload track file: {track_file}, count={count}')
        total += count
    print(f'uploaded: total count = {total}')

def update_tracklet(args_conf, store:TrackletStore) -> None:
    store.insert_or_update_tracklet(args_conf.node_id, args_conf.track_id)


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    conf, db_conf, args_conf = load_node_conf(args)
    
    store = TrackletStore(sql_utils.SQLConnector.from_conf(db_conf))
    if args_conf.subparsers == 'format':
        format(args_conf, store)
    elif args_conf.subparsers == 'drop':
        store.drop()
    elif args_conf.subparsers == 'upload':
        upload(args_conf, store)
    
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