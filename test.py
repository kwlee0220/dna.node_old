
from kafka import KafkaConsumer

import numpy as np
import os, json, cv2, random

from dna.node import TrackEvent


consumer = KafkaConsumer('test',
                        bootstrap_servers=['localhost:9092'],
                        auto_offset_reset='earliest')
consumer.subscribe('track_events')

for msg in consumer:
    key = msg.key.decode('utf-8')
    value = msg.value.decode('utf-8')
    ev:TrackEvent = TrackEvent.from_json(value)


    print(f'Topic: {msg.topic}, Offset: {msg.offset}, Event: {ev}')