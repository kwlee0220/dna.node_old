
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
    parser = argparse.ArgumentParser(description="Draw paths")
    parser.add_argument("track_files", nargs='+', help="track json files")
    parser.add_argument("--batch", metavar="count", type=int, default=30, help="upload batch count")
    
    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_dbname", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()


def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    conf, db_conf, args_conf = load_node_conf(args)
    
    store = TrackletStore(sql_utils.SQLConnector.from_conf(db_conf))
    
    # total = 0
    # for track_file in args.track_files:
    #     count = store.insert_tracks(read_tracks_json(track_file))
    #     print(f'upload track file: {track_file}, count={count}')
    #     total += count
    # print(f'uploaded: total count = {total}')
    
    first, last = store.read_first_and_last_track(node_id='etri:04', track_id='1')
    print(first)
    print(last)

if __name__ == '__main__':
    main()