
# from typing import Tuple, List, Dict, Union, Generator, Set, Iterable
from contextlib import closing
from datetime import timedelta

from omegaconf import OmegaConf
from kafka import KafkaConsumer

from dna import initialize_logger, config
from dna.node import TrackEvent, TrackFeature, EventListener
from dna.node.event_processors import PrintEvent
from dna.assoc.tracklet_store import TrackletStore
from dna.assoc import Association, AssociationCollection, AssociationCollector
from dna.assoc.associator_motion import NodeAssociationSchema, MotionBasedTrackletAssociator
from dna.assoc.associator_feature import FeatureBasedTrackletAssociator
from dna.assoc.closure import AssociationClosureBuilder, Extend
from dna.assoc.utils import FixedIntervalCollector
from dna.support.sql_utils import SQLConnector

import logging
LOGGER = logging.getLogger('dna.assoc_tracklets')


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Tracklet and tracks commands")
    
    parser.add_argument("node_pairs", nargs='+', help="target node ids")
    parser.add_argument("--boostrap_servers", default=['localhost:9092'], help="kafka server")
    parser.add_argument("--listen", nargs='+', help="listening nodes")
    parser.add_argument("--max_distance_to_camera", type=float, metavar="meter", default=55,
                        help="max. distance from camera (default: 55)")
    parser.add_argument("--max_track_distance", type=float, metavar="meter", default=5,
                        help="max. distance between two locations (default: 5)")
    parser.add_argument("--window_interval", type=float, metavar="seconds", default=1,
                        help="window interval seconds (default: 1)")
    parser.add_argument("--idle_timeout", type=float, metavar="seconds", default=10000,
                        help="idle timeout seconds (default: 1)")
    
    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_dbname", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")
    
    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")

    return parser.parse_known_args()

def consume_tracks_upto(consumer:KafkaConsumer, listener:EventListener, upto_ms:int):
    last_ts = 0
    while True:
        partitions = consumer.poll(timeout_ms=1000, max_records=50)
        if partitions:
            for topic_info, partition in partitions.items():
                if topic_info.topic == 'track-events':
                    for serialized in partition:
                        ev = TrackEvent.deserialize(serialized.value)
                        listener.handle_event(ev)
                        last_ts = ev.ts
            if last_ts > upto_ms:
                return last_ts
        else:
            return None
    
def consume_features_upto(consumer:KafkaConsumer, listening_nodes, listener:EventListener, upto_ms:int):
    while True:
        partitions = consumer.poll(timeout_ms=1000, max_records=10)
        if partitions:
            for topic_info, partition in partitions.items():
                if topic_info.topic == 'track-features':
                    for serialized in partition:
                        ev = TrackFeature.deserialize(serialized.value)
                        if ev.zone_relation == 'D':
                            listener.handle_event(ev)
                        elif serialized.key in listening_nodes:
                            listener.handle_event(ev)
                        last_ts = ev.ts
            if last_ts > upto_ms:
                return last_ts
        else:
            return None

def main():
    args, _ = parse_args()

    initialize_logger(args.logger)
    logger = logging.getLogger('dna.assoc.motion')
    
    # argument에 기술된 conf를 사용하여 configuration 파일을 읽는다.
    conf = config.to_conf(args)
    
    assoc_pairs = [tuple(pair_str.split('-')) for pair_str in args.node_pairs]
    schema = NodeAssociationSchema(assoc_pairs)
        
    window_interval_ms = round(args.window_interval * 1000)
    motion_associator = MotionBasedTrackletAssociator(schema,
                                               window_interval_ms=window_interval_ms,
                                               max_distance_to_camera=args.max_distance_to_camera,
                                               max_track_distance=args.max_track_distance,
                                               idle_timeout=args.idle_timeout)
    
    motion_closures = AssociationCollection(keep_best_association_only=True)
    motion_closure_builder = AssociationClosureBuilder(collection=motion_closures)
    motion_associator.add_listener(motion_closure_builder)
    
    #######################################################################################################
    
    store = TrackletStore(SQLConnector.from_conf(conf))
    feature_associator = FeatureBasedTrackletAssociator(store=store,
                                              listen_nodes=args.listen,
                                              prefix_length=5,
                                              top_k=4,
                                              logger=LOGGER.getChild('associator'))
    
    interval_collector = FixedIntervalCollector(interval_ms=3*1000)
    feature_associator.add_listener(interval_collector)
    
    feature_closures = AssociationClosureBuilder()
    interval_collector.add_listener(feature_closures)
    
    #######################################################################################################
    
    extend = Extend(motion_closures)
    feature_closures.add_listener(extend)
    final_collector = AssociationCollector()
    extend.add_listener(final_collector)
    extend.add_listener(PrintEvent("**** "))
    
    #######################################################################################################
    
    motion_associator.start()
    
    bootstrap_servers = config.get(conf, 'bootstrap_servers', default=['localhost:9092'])
    offset = config.get(conf, 'auto_offset_reset', default='earliest')
    event_consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                             auto_offset_reset=offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    event_consumer.subscribe(['track-events'])
    feature_consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers,
                             auto_offset_reset=offset,
                             key_deserializer=lambda k: k.decode('utf-8'))
    feature_consumer.subscribe(['track-features'])
    
    listening_nodes = set(args.listen)
    last_ts = 0
    while True:
        last_ts = consume_tracks_upto(event_consumer, motion_associator, last_ts)
        if last_ts is None: break
        last_ts = consume_features_upto(feature_consumer, listening_nodes, feature_associator, last_ts)
        if last_ts is None: break
    
    motion_associator.close()
    feature_associator.close()
    
    for assoc in motion_closures:
        print(assoc)
     
if __name__ == '__main__':
    main()