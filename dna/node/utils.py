from typing import Any, Optional
from dataclasses import dataclass, field

from pubsub import PubSub, Queue


class EventPublisher:
    __slots__ = 'pubsub', 'channel', 'nsubscribers'

    def __init__(self, pubsub: PubSub, channel: str) -> None:
        self.pubsub = pubsub
        self.channel = channel
        self.nsubscribers = 0

    def publish(self, ev: Any) -> None:
        if self.nsubscribers > 0:
            self.pubsub.publish(self.channel, ev)

    def subscribe(self) -> Queue:
        self.nsubscribers += 1
        return self.pubsub.subscribe(self.channel)

    def __repr__(self) -> str:
        return f'EventPublisher[channel={self.channel}]'