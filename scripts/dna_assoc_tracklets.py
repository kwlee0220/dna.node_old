
# from typing import Tuple, List, Dict, Union, Generator, Set, Iterable

from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import initialize_logger
from dna.config import load_node_conf2
from dna.node.tracklet_store import TrackletStore
from dna.support.sql_utils import SQLConnector
from dna.assoc import FeatureBasedTrackletAssociator


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Tracklet and tracks commands")
    
    parser.add_argument("--boostrap_servers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--listen", nargs='+', help="listening nodes")
    
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
    conf, db_conf, args_conf = load_node_conf2(args)
    
    store = TrackletStore(SQLConnector.from_conf(db_conf))
    consumer = KafkaConsumer(bootstrap_servers=args_conf.boostrap_servers,
                             auto_offset_reset='earliest',
                             key_deserializer=lambda k: k.decode('utf-8'))
    consumer.subscribe(['track-features'])
    associator = FeatureBasedTrackletAssociator(consumer, store, args.listen)
    associator.run()
                            
            
if __name__ == '__main__':
    main()