from __future__ import annotations

from .event_processor import EventProcessor
from .event_processors import EventRelay


class MultiStagePipeline(EventProcessor):
    def __init__(self) -> None:
        super().__init__()
        
        self.end_of_stage = EventRelay(self)
        self.stages:list[EventProcessor] = []

    def close(self) -> None:
        super().close()
        for stage in reversed(self.stages):
            stage.close()

    def add_stage(self, proc:EventProcessor) -> None:
        if self.stages:
            last_stage = self.stages[-1]
            last_stage.remove_listener(self.end_of_stage)
            last_stage.add_listener(proc)
        self.stages.append(proc)
        proc.add_listener(self.end_of_stage)
            
    def handle_event(self, ev:object) -> None:
        self.stages[0].handle_event(ev)
        
    def __repr__(self) -> str:
        return '[' + " -> ".join(repr(stage) for stage in self.stages) + ']'