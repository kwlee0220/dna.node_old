# vim: expandtab:ts=4:sw=4

import numpy as np

from dna.detect import Detection
from dna.tracker.dna_deepsort.deepsort import utils

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    STATE_NAME = ['NONE', 'Tentative', 'Confirmed', 'Deleted']
    STATE_ABBR = ['N', 'T', 'C', 'D']


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, detection: Detection=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        self.last_detection = detection
        if detection is not None and detection.feature is not None:
            self.features.append(detection.feature)

        self.time_to_promote = n_init-1
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x, max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection: Detection, track_params) -> None:
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.bbox.to_xyah())
        if utils.is_large_detection_for_metric(detection):
            self.features.append(detection.feature)
            if len(self.features) > track_params.max_feature_count:
                self.features = self.features[-track_params.max_feature_count:]
        self.last_detection = detection
        
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative:
            if detection.score >= track_params.detection_threshold:
                self.time_to_promote -= 1
                if self.time_to_promote == 0:
                    self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        self.last_detection = None

    # kwlee
    def mark_deleted(self):
        self.state = TrackState.Deleted
        self.last_detection = None

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    # kwlee
    def __repr__(self) -> str:
        state_str = TrackState.STATE_NAME[self.state]
        promote_str = f', ttp={self.time_to_promote}' if self.state == TrackState.Tentative else ""
        tlwh = self.to_tlwh()
        return (f"{state_str}[{self.track_id}, age={self.age}({self.time_since_update}){promote_str}, "
                f"loc=({tlwh[0]:.0f},{tlwh[1]:.0f}):{tlwh[2]:.0f}x{tlwh[3]:.0f}")
        
    # kwlee
    @property
    def short_repr(self) -> str:
        if self.state == TrackState.Confirmed:
            if self.time_since_update == 0:
                return f"{self.track_id}(C)"
            else:
                return f"{self.track_id}({self.time_since_update})"
        elif self.state == TrackState.Tentative:
            return f"{self.track_id}(-{self.time_to_promote})"
        elif self.state == TrackState.Deleted:
            return f"{self.track_id}(D)"
        else:
            raise ValueError("Shold not be here")