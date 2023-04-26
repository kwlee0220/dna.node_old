from __future__ import annotations

from typing import List, Dict, Optional, Union

from omegaconf import OmegaConf
import numpy as np

from dna import Point
from dna.node import TrackEvent, TimeElapsed, EventProcessor
from .types import KafkaEvent


ALPHA = 1   # smoothing hyper parameter
            # ALPHA가 강할 수록 smoothing을 더 강하게 진행 (기존 location 정보를 잃을 수도 있음)

def smoothing_track(track, alpha=ALPHA):
    """

    Parameters
    ----------
    track: 입력 받은 track 정보 (location 정보)
    alpha: smoothing hyper parameter

    Returns: stabilization을 완료한 track location 정보를 반환
    -------

    """
    l = len(track)
    ll = l * 3 - 3  # l + (l - 1) + (l - 2) matrix size
    A = np.zeros((ll, l))
    A[:l, :] = np.eye(l)
    A[l:l * 2 - 1, :] = alpha * (np.eye(l) - np.eye(l, k=1))[:l - 1, :]  # l Plot1
    A[l * 2 - 1:, :] = alpha * (2 * np.eye(l) - np.eye(l, k=1) - np.eye(l, k=-1))[1:l - 1, :]  # l - 2

    b = np.zeros((1, ll))
    b[:, :l] = track

    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b.T)
    X = np.dot(np.linalg.inv(ATA), ATb)
    return X


def stabilization_location(location, frame=5, alpha=ALPHA):
    """
    Parameters
    ----------
    location: trajectory의 위치 정보
    frame: 앞 뒤로 몇 프레임까지 볼 것인지.

    Returns: 안정화된 위치 정보
    -------
    """
    stab_location = []
    coord_length = len(location)
    for idx, coord in enumerate(location):
        if idx < frame and idx + frame < coord_length:
            # len(prev information) < frame
            # 과거 정보가 부족한 경우
            prev_locations = location[:idx + 1]  # prev location + current location
            frame_ = len(prev_locations)
            next_locations = location[idx + 1:idx + 1 + frame_]
            smoothing_coord = smoothing_track(np.concatenate([prev_locations, next_locations]), alpha=alpha)[idx][0]
        elif idx < coord_length - frame and idx - frame >= 0:
            # len(next information) >= frame and len(prev information) >= frame
            prev_locations = location[idx - frame:idx + 1]  # prev location + current location
            next_locations = location[idx + 1:idx + 1 + frame]
            smoothing_coord = smoothing_track(np.concatenate([prev_locations, next_locations]), alpha=1)[frame][0]
            # 과거 정보, 미래 정보 모두 있는 경우
        elif idx - frame >= 0:
            # len(next information) < frame
            # 미래 정보가 부족한 경우
            next_locations = location[idx + 1:]
            frame_ = len(next_locations)
            prev_locations = location[idx - frame_:idx + 1]  # prev location + current location
            if len(np.concatenate([prev_locations, next_locations])) == 1:
                smoothing_coord = location[idx]
            else:
                smoothing_coord = \
                smoothing_track(np.concatenate([prev_locations, next_locations]), alpha=alpha)[-(frame_ + 1)][0]
        else:
            # Short frame
            # parameter로 받은 location 정보 자체가 짧은 경우
            if len(location[:idx + 1]) == 1:
                smoothing_coord = location[idx]
            else:
                smoothing_coord = smoothing_track(location[:idx + 1], alpha=alpha)[-1][0]
        stab_location.append(smoothing_coord)
    return stab_location

_DEFAULT_SMOOTHING_FACTOR = 1
class Stabilizer(EventProcessor):
    def __init__(self, conf:OmegaConf) -> None:
        EventProcessor.__init__(self)
        self.look_ahead = conf.look_ahead
        self.smoothing_factor = conf.get("smoothing_factor", 1)
        self.current, self.upper = 0, 0

        self.pending_events: List[TrackEvent] = []
        self.pending_xs: List[float] = []
        self.pending_ys: List[float] = []

    def close(self) -> None:
        xs = stabilization_location(self.pending_xs, self.look_ahead)
        ys = stabilization_location(self.pending_ys, self.look_ahead)
        for i in range(len(self.pending_events)):
            if i >= self.current:
                pt = Point(x=xs[i], y=ys[i])
                ev = self.pending_events[i]
                stabilized = ev.updated(world_coord=pt)
                # print(f"{stabilized.frame_index}: {ev.world_coord} -> {stabilized.world_coord}")
                self._publish_event(stabilized)

        super().close()
        
    def min_frame_index(self) -> int:
        return self.pending_events[0].frame_index if self.pending_events else None
        
    def handle_event(self, ev:TrackEvent|TimeElapsed) -> None:
        if isinstance(ev, TrackEvent):
            self.pending_events.append(ev)
            x, y = tuple(ev.world_coord.xy)
            self.pending_xs.append(x)
            self.pending_ys.append(y)
            self.upper += 1

            if self.upper - self.current > self.look_ahead:
                xs = stabilization_location(self.pending_xs, self.look_ahead)
                ys = stabilization_location(self.pending_ys, self.look_ahead)

                pt = Point(x=xs[self.current], y=ys[self.current])
                ev = self.pending_events[self.current]
                stabilized = ev.updated(world_coord=pt)
                # print(f"{stabilized.frame_index}: {ev.world_coord} -> {stabilized.world_coord}")
                self._publish_event(stabilized)

                self.current += 1
                if self.current > self.look_ahead:
                    self.pending_events.pop(0)
                    self.pending_xs.pop(0)
                    self.pending_ys.pop(0)
                    self.current -= 1
                    self.upper -= 1
        else:
            self._publish_event(ev)


