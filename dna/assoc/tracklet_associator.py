from __future__ import annotations
from typing import List, Union, Set, Dict
from dataclasses import dataclass

import numpy as np
from kafka import KafkaConsumer

from dna.node import TrackFeature
from dna.node.zone import ZoneRelation
# from dna.node.tracklet_store import TrackletStore
from dna.tracker.utils import cosine_distance
from .types import TrackletId, Association


class IncomingLink:
    __slots__ = ('node_id', 'enter_zone', 'exit_zone', 'transition_time', 'node_travel_time')
    
    def __init__(self, from_node_id:str, enter_zone:str, exit_zone:str,
                 transition_time:float, node_travel_time:float) -> None:
        self.node_id = from_node_id
        self.enter_zone = enter_zone
        self.exit_zone = exit_zone
        self.transition_time = transition_time
        self.node_travel_time = node_travel_time
        
    @property
    def transition_millis(self):
        return round(self.transition_time * 1000)
        
    @property
    def node_travel_millis(self):
        return round(self.node_travel_time * 1000)
        
    def __eq__(self, other: object) -> bool:
        if other and isinstance(other, IncomingLink):
            return self.node_id == other.node_id  \
                    and self.enter_zone == other.enter_zone \
                    and self.exit_zone == other.exit_zone
        else:
            return False
        
    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
        
    def __hash__(self) -> int:
        return hash(self.node_id, self.enter_zone, self.exit_zone)
    
    def __repr__(self) -> str:
        enter_zone_str = self.enter_zone if self.enter_zone else '*'
        exit_zone_str = self.exit_zone if self.exit_zone else '*'
        return ( f'{self.node_id}[{enter_zone_str}->{exit_zone_str}], '
                f'{self.node_travel_millis:.3f}, {self.transition_millis:.3f}' )
    

NODE_NETWORK:Dict[str,Dict[str,List[IncomingLink]]] = {
    'etri:04': {
        'C': [ IncomingLink('etri:05', None, 'C', 0, 3) ],
    },
    'etri:05': {
        'A': [IncomingLink('etri:07', None, 'A', 0, 3),
              IncomingLink('etri:06', 'A', None, 0, 3),
              IncomingLink('etri:04', 'A', None, 0, 3),], 
        'B': [IncomingLink('etri:04', 'B', None, -3, 5),
              IncomingLink('etri:06', 'B', None, -1, 3), ],
        'C': [IncomingLink('etri:04', 'C', None, 0, 3)],
    },
    'etri:06': {
    },
    'etri:07': {
        'A': [ IncomingLink('etri:05', None, 'A', 0, 3) ],
    }
}


class AssociationSession:
    __slots__ = ('id', 't_features', 'enter_zone', 'association')
    
    def __init__(self, id:TrackletId) -> None:
        self.id = id
        self.t_features:List[TrackFeature] = []
        self.enter_zone = None
        self.association:Association = None
        
    def append(self, t_feature:TrackFeature) -> None:
        self.t_features.append(t_feature)
        if not self.enter_zone:
            zone_rel, zone_id = ZoneRelation.parseRelationStr(t_feature.zone_relation)
            match zone_rel:
                case ZoneRelation.Entered | ZoneRelation.Inside:
                    self.enter_zone = zone_id
    
    def __len__(self) -> int:
        return len(self.t_features)
    
    def __iter__(self):
        return (t_feature for t_feature in self.t_features)
    
    def __getitem__(self, index) -> Union[TrackFeature, List[TrackFeature]]:
        return self.t_features[index]
                    
    def features(self) -> np.ndarray:
        return np.array([tf.feature for tf in self.t_features])
    
    def __repr__(self) -> str:
        peer = f', peer: {self.association.tracklet2} ({self.association.distance:.3f})' if self.association else ''
        return f'{self.id}, enter={self.enter_zone}, n_features=[{len(self.t_features)}]{peer}'


class FeatureBasedTrackletAssociator:
    __slots__ = ( 'consumer', 'store', 'target_nodes', 'prefix_length', 'association_threshold', 'associations', 'top_k' )
    
    def __init__(self, consumer:KafkaConsumer, store:TrackletStore, target_nodes:Set[str],
                 /, prefix_length:int=999, association_threshold:float=0.25) -> None:
        self.consumer = consumer
        self.store = store
        self.target_nodes = target_nodes
        self.prefix_length = prefix_length
        self.association_threshold = association_threshold
        self.associations:Set[Association] = set()
        self.top_k:int = 3
        
    def run(self) -> None:
        already_matcheds:Set[TrackletId] = set()
        sessions:Dict[TrackletId,AssociationSession] = dict()
        
        self.consumer.subscribe(['track-features'])
        while True:
            partitions = self.consumer.poll(timeout_ms=500, max_records=100)  
            for messages in partitions.values():
                for msg in messages:
                    if msg.key not in self.target_nodes:
                        continue
                    
                    t_feature = TrackFeature.deserialize(msg.value)
                    tracklet_id = TrackletId(t_feature.node_id, t_feature.track_id)
                    
                    if t_feature.zone_relation == 'D':
                        if tracklet_id in already_matcheds:
                            already_matcheds.discard(tracklet_id)
                        else:
                            session = sessions.pop(tracklet_id, None)
                            if session and session.enter_zone:
                                self.associate(session)
                    elif tracklet_id not in already_matcheds:
                        session = self.get_or_create_session(sessions, tracklet_id)
                        session.append(t_feature)
                        n_extends = len(session) - self.prefix_length
                        if (n_extends >= 0 and (n_extends % 5) == 0) and session.enter_zone:
                            if self.associate(session):
                                session = sessions.pop(tracklet_id)
                                already_matcheds.add(tracklet_id)
                            else:
                                print(f'wait for futher features: {session}')

    def associate(self, session:AssociationSession) -> bool:
        print(f'{session}')
        
        target_id = session.id
        # if self.store.list_fragments_of_tracklet(session.id):
        #     return True
        
        target_node_links = NODE_NETWORK.get(session.id.node_id, dict())
        incoming_links = target_node_links.get(session.enter_zone, [])
        
        for incoming_link in incoming_links:
            incoming_node = incoming_link.node_id
            end_ts = session[0].ts - incoming_link.transition_millis
            begin_ts = end_ts - incoming_link.node_travel_millis
            time_based_candidates = set(self.store.list_tracklet_range(incoming_node, begin_ts, end_ts))
            if not time_based_candidates:
                # incoming node에 예상되는 시간 구간의 tracklet이 존재하지 않는다면
                # association을 시도할 여지가 없음.
                continue
            
            incoming_candidates = self.store.list_tracklet_metas(incoming_node,
                                                                 enter_zone=incoming_link.enter_zone,
                                                                 exit_zone=incoming_link.exit_zone)
            incoming_candidates = {tracklet_meta.track_id for tracklet_meta in incoming_candidates}
            
            ignores = {assoc.tracklet2.track_id for assoc in self.associations
                                                    if assoc.tracklet2.node_id == incoming_node}
            
            # incoming node에서 검출된 tracklet들 중에서 시간 조건과 exit_zone 조건을 고려하여 최종적으로
            # association 대상 tracklet을 찾는다
            candidates = (time_based_candidates & incoming_candidates) #- ignores
            
            best_match = (None, 999)
            for track_id in candidates:
                peer_id = TrackletId(incoming_node, track_id)
                score = self.calc_match_score(session, peer_id)
                if score < best_match[1]:
                    best_match = (peer_id, score)
            if best_match[0] and best_match[1] < self.association_threshold:
                session.association = Association(tracklet1=target_id, tracklet2=best_match[0], distance=best_match[1])
                # self.store.insert_association(session.association)
                self.associations.add(session.association)
                print(f"\tassociation: {session.association}")
                pass
        
    def get_or_create_session(self, sessions, tracklet_id:TrackletId) -> AssociationSession:
        session = sessions.get(tracklet_id)
        if not session:
            session = AssociationSession(tracklet_id)
            sessions[tracklet_id] = session
        return session
    
    def calc_match_score(self, session:AssociationSession, peer_tracklet:TrackletId):
        track_features = self.store.read_track_features(peer_tracklet.node_id, peer_tracklet.track_id)
        if len(track_features) > 0:
            features = np.array([t_f.feature for t_f in track_features])
            distances = cosine_distance(session.features(), features, True).min(axis=0)
            
            if len(distances) > self.top_k:
                topk_dists = np.partition(distances, self.top_k)[:self.top_k]
            else:
                topk_dists = distances
            distance = np.mean(topk_dists)
            
            distances = np.around(np.sort(topk_dists), decimals=3)
            peer_id = TrackletId(track_features[0].node_id, track_features[0].track_id)
            print(f"\t{peer_id}: {distance:.3f} ({distances})")
            
            return distance
        else:
            return 999
