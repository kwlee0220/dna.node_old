from __future__ import annotations

from typing import List, Dict
from dataclasses import dataclass, field

from pubsub import PubSub, Queue

from dna.track import TrackState, Track
from .track_event import TrackEvent
from .event_processor import EventProcessor
from .utils import EventPublisher


@dataclass(eq=True, slots=True)
class Session:
    id: str = field(hash=True)
    state: TrackState = field(hash=False, compare=False)
    pendings: List[TrackEvent] = field(hash=False, compare=False)


class RefineTrackEvent(EventProcessor):
    __slots__ = ('sessions', )

    def __init__(self, in_queue: Queue, publisher: EventPublisher) -> None:
        super().__init__(in_queue, publisher)

        self.sessions: Dict[str, Session] = {}

    def handle_event(self, ev) -> None:
        session: Session = self.sessions.get(ev.luid, None)
        if ev.state == TrackState.Deleted:   # tracking이 종료된 경우
            self.sessions.pop(ev.luid, None)
            if session.state != TrackState.Tentative:
                self.publish_event(ev)
        else:
            if session is None: # TrackState.Null or TrackState.Deleted
                self.__on_initial(ev)
            elif session.state == TrackState.Confirmed:
                self.__on_confirmed(session, ev)
            elif session.state == TrackState.Tentative:
                self.__on_tentative(session, ev)
            elif session.state == TrackState.TemporarilyLost:
                self.__on_temporarily_lost(session, ev)

    def __on_initial(self, ev: TrackEvent) -> None:
        # track과 관련된 session 정보가 없다는 것은 이 track event가 한 물체의 첫번째 track event라는
        # 것을 의미하기 때문에 session을 새로 생성한다.
        if ev.state == TrackState.Tentative:
            self.sessions[ev.luid] = Session(id=ev.luid, state=TrackState.Tentative, pendings=[ev])
        elif ev.state == TrackState.Confirmed:
            self.sessions[ev.luid] = Session(id=ev.luid, state=TrackState.Confirmed, pendings=[])
        else:
            raise AssertionError(f"unexpected track event (invalid track state): {ev.track}")

    def __on_confirmed(self, session: Session, ev: TrackEvent) -> None:
        if ev.state == TrackState.Confirmed:
            self.publish_event(ev)
        elif ev.state == TrackState.TemporarilyLost:
            session.pendings.append(ev)
            session.state = TrackState.TemporarilyLost

    def __on_tentative(self, session: Session, ev: TrackEvent) -> None:
        if ev.state == TrackState.Confirmed:
            # 본 trail을 임시 상태에서 정식의 trail 상태로 변환시키고,
            # 지금까지 pending된 모든 tentative event를 trail에 포함시킨다
            # self.logger.debug(f"accept tentative tracks: track={track.id}, count={len(session.pendings)}")
            self.__publish_all_pended_events(session)
            self.publish_event(ev)
            session.state = TrackState.Confirmed
        elif ev.state == TrackState.Tentative:
            session.pendings.append(ev)

    def __on_temporarily_lost(self, session: Session, ev: TrackEvent) -> None:
        if ev.state == TrackState.Confirmed:
            # self.logger.debug(f"accept 'temporarily-lost' tracks[{ev.luid}]: count={len(session.pendings)}")

            self.__publish_all_pended_events(session)
            self.publish_event(ev)
            session.state = TrackState.Confirmed
        elif ev.state == TrackState.TemporarilyLost:
            session.pendings.append(ev)

    def __publish_all_pended_events(self, session):
        for pended in session.pendings:
            self.publish_event(pended)
        session.pendings.clear()