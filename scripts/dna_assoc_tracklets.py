
# from typing import Tuple, List, Dict, Union, Generator, Set, Iterable
from contextlib import closing
from datetime import timedelta

from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import initialize_logger, config
from dna.node import TrackEvent, TrackFeature
from dna.node.event_processors import PrintEvent
from dna.assoc.tracklet_store import TrackletStore
from dna.support.sql_utils import SQLConnector
from dna.assoc.associator_feature import FeatureBasedTrackletAssociator
from dna.assoc.utils import FixedIntervalCollector, AssociationCloser, FixedIntervalClosureBuilder, ClosedTrackletCollector
from dna.assoc import AssociationCollection, AssociationCollector, AssociationClosureBuilder

import logging
LOGGER = logging.getLogger('dna.assoc.features')


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Tracklet and tracks commands")
    
    parser.add_argument("--boostrap_servers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--listen", nargs='+', help="listening nodes")
    parser.add_argument("--motion_score_threshold", type=float, metavar="0~1", default=0.35,
                        help="motion match score threshold (default: 0.35)")
    
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
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    store = TrackletStore(SQLConnector.from_conf(conf))
    associator = FeatureBasedTrackletAssociator(store=store,
                                              listen_nodes=args.listen,
                                              prefix_length=5,
                                              top_k=4,
                                              logger=LOGGER.getChild('associator'))
    
    close_collector = ClosedTrackletCollector()
    associator.add_listener(close_collector)
    
    interval_collector = FixedIntervalClosureBuilder(interval_ms=5*1000,
                                                     closer_collector=close_collector,
                                                     logger=LOGGER.getChild('interval'))
    associator.add_listener(interval_collector)
    
    closures = AssociationCollection(keep_best_association_only=True)
    closuer_builder = AssociationClosureBuilder(collection=closures)
    closuer_builder.add_listener(PrintEvent("**** "))
    interval_collector.add_listener(closuer_builder)
    
    
    bootstrap_servers = config.get(conf, 'bootstrap_servers', default=['localhost:9092'])
    offset = config.get(conf, 'auto_offset_reset', default='earliest')
    feature_consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                                     auto_offset_reset=offset,
                                     key_deserializer=lambda k: k.decode('utf-8'))
    feature_consumer.subscribe(['track-features'])
    
    listening_nodes = set(args.listen)
    while True:
        partitions = feature_consumer.poll(timeout_ms=3000, max_records=10)
        if partitions:
            for topic_info, partition in partitions.items():
                match topic_info.topic:
                    case 'track-features':
                        for key, feature in ((serialized.key, TrackFeature.deserialize(serialized.value)) for serialized in partition):
                            if key in listening_nodes or feature.zone_relation == 'D':
                                associator.handle_event(feature)
        else:
            break
          
    associator.close()
    for assoc in closures:
        print(assoc)                  
            
if __name__ == '__main__':
    main()