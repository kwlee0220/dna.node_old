from __future__ import annotations

from abc import ABCMeta, abstractmethod


class KafkaEvent(metaclass=ABCMeta):
    @abstractmethod
    def key(self) -> str: pass
    
    @abstractmethod
    def serialize(self) -> str: pass