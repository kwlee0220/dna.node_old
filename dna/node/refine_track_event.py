from __future__ import annotations

from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from dna.tracker.dna_track import TrackState
from .track_event import TrackEvent
from .event_processor import EventProcessor

import logging
LOGGER = logging.getLogger("dna.node.event")


@dataclass(eq=True)    # slots=True
class Session:
    id: int = field(hash=True)
    state: TrackState = field(hash=False, compare=False)
    pendings: List[TrackEvent] = field(hash=False, compare=False)

    def find_insert_index(self, frame_index:int) -> int:
        gap = self.pendings[-1].frame_index - frame_index + 1
        if gap > 0:
            return len(self.pendings) - (gap + 1)
        else:
            return -1

    def trim_right_to(self, frame_index:int) -> None:
        trim_offset = self.pendings[-1].frame_index - frame_index + 1
        if trim_offset > 0:
            offset = len(self.pendings) - trim_offset
            self.pendings = self.pendings[:offset]
            
    def __repr__(self) -> str:
        interval_str = ""
        if self.pendings:
            first = self.pendings[0].frame_index
            last = self.pendings[-1].frame_index
            interval_str = f':{first}-{last}'
        return f'{self.id}({self.state.abbr})[{len(self.pendings)}{interval_str}]'


class RefineTrackEvent(EventProcessor):
    __slots__ = ('sessions', 'max_pendings')

    def __init__(self, max_pendings:int=30) -> None:
        EventProcessor.__init__(self)

        self.sessions: Dict[int, Session] = {}
        self.max_pendings = max_pendings

    def close(self) -> None:
        self.sessions.clear()
        super().close()

    def handle_event(self, ev:TrackEvent) -> None:
        session:Session = self.sessions.get(ev.track_id, None)
        if ev.state == TrackState.Deleted:   # tracking이 종료된 경우
            if session:
                self.__on_delete_event(session, ev)
        else:
            if session is None: # TrackState.Null or TrackState.Deleted
                self.__on_initial(ev)
            elif session.state == TrackState.Confirmed:
                self.__on_confirmed(session, ev)
            elif session.state == TrackState.Tentative:
                self.__on_tentative(session, ev)
            elif session.state == TrackState.TemporarilyLost:
                self.__on_temporarily_lost(session, ev)

    def __on_initial(self, ev:TrackEvent) -> None:
        # track과 관련된 session 정보가 없다는 것은 이 track event가 한 물체의 첫번째 track event라는 것을 
        # 의미하기 때문에 session을 새로 생성한다.
        self.sessions[ev.track_id] = Session(id=ev.track_id, state=ev.state, pendings=[])
        if ev.state == TrackState.Tentative:
            self.sessions[ev.track_id].pendings.append(ev)
        elif ev.state == TrackState.Confirmed:
            self.publish_event(ev)
        else:
            raise AssertionError(f"unexpected track event (invalid track state): {ev.track}")

    def __on_delete_event(self, session, ev:TrackEvent) -> None:
        # remove this session.
        self.sessions.pop(ev.track_id, None)

        if session.state == TrackState.Tentative:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"discard all pending tentative track events: track_id={ev.track_id}, count={len(session.pendings)}")
        elif session.state == TrackState.TemporarilyLost:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"discard all pending lost track events: track_id={ev.track_id}, count={len(session.pendings)}")
        self.publish_event(ev)

    def __on_confirmed(self, session:Session, ev:TrackEvent) -> None:
        if ev.state == TrackState.Confirmed:
            self.publish_event(ev)
        elif ev.state == TrackState.TemporarilyLost:
            session.pendings.append(ev)
            session.state = TrackState.TemporarilyLost
        else:
            raise AssertionError(f"unexpected track event (invalid track state): state={session.state}, event={ev.track}")

    def __on_tentative(self, session:Session, ev:TrackEvent) -> None:
        if ev.state == TrackState.Confirmed:
            # 본 trail을 임시 상태에서 정식의 trail 상태로 변환시키고,
            # 지금까지 pending된 모든 tentative event를 trail에 포함시킨다
            # self.logger.debug(f"accept tentative tracks: track={track.id}, count={len(session.pendings)}")
            self.__publish_all_pended_events(session)
            self.publish_event(ev)
            session.state = TrackState.Confirmed
        elif ev.state == TrackState.Tentative:
            session.pendings.append(ev)
        else:
            raise AssertionError(f"unexpected track event (invalid track state): state={session.state}, event={ev.track}")

    def __on_temporarily_lost(self, session:Session, ev: TrackEvent) -> None:
        if ev.state == TrackState.Confirmed:
            session.trim_right_to(ev.frame_index)
            self.__publish_all_pended_events(session)
            self.publish_event(ev)
            session.state = TrackState.Confirmed
        elif ev.state == TrackState.TemporarilyLost:
            insert_index = session.find_insert_index(ev.frame_index)
            if insert_index < 0:
                session.pendings.append(ev)
                while len(session.pendings) > self.max_pendings:
                    self.publish_event(session.pendings.pop(0))

    def __publish_all_pended_events(self, session):
        for pended in session.pendings:
            self.publish_event(pended)
        session.pendings.clear()