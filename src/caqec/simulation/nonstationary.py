"""Non-stationary bit-flip noise: schedules and stream generation.

SIMULATOR-ONLY (hidden from any online controller): the schedule, p over time, the change time,
the labels y and the raw noise draws. They live in GroundTruth. An online method may only use
StreamBatch.observable, the syndrome features X, and only shot by shot, in time order."""
from dataclasses import dataclass

import numpy as np

from caqec.simulation.dataset import rows_from_realization

KINDS = ("stationary", "abrupt", "gradual", "piecewise")


@dataclass(frozen=True)
class NoiseSchedule:
    kind: str = "stationary"
    p0: float = 0.05
    p1: float = 0.05          # abrupt/gradual: the value after the change
    change_time: int = 0      # abrupt: step time; gradual: ramp start
    ramp_len: int = 1         # gradual: shots to ramp from p0 to p1
    steps: tuple = ()         # piecewise: ((time, p), ...) in increasing time; p0 holds before the first

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        probs = [self.p0, self.p1] + [p for _, p in self.steps]
        if not all(0.0 <= p <= 1.0 for p in probs):
            raise ValueError("all probabilities must be in [0, 1]")
        if self.change_time < 0 or self.ramp_len < 1:
            raise ValueError("change_time must be >= 0 and ramp_len >= 1")
        times = [t for t, _ in self.steps]
        if times != sorted(times) or any(t < 0 for t in times):
            raise ValueError("piecewise steps must have non-negative, increasing times")

    def p_at(self, T: int) -> np.ndarray:
        t = np.arange(T)
        if self.kind == "stationary":
            return np.full(T, self.p0)
        if self.kind == "abrupt":
            return np.where(t < self.change_time, self.p0, self.p1).astype(float)
        if self.kind == "gradual":
            frac = np.clip((t - self.change_time) / self.ramp_len, 0.0, 1.0)
            return self.p0 + (self.p1 - self.p0) * frac
        p = np.full(T, self.p0)
        for time, p_new in self.steps:
            p[t >= time] = p_new
        return p

    @property
    def change_start(self):
        """First shot at which the noise differs from p0 (None if stationary). Ground truth."""
        if self.kind == "stationary":
            return None
        if self.kind == "piecewise":
            return self.steps[0][0] if self.steps else None
        return self.change_time

    @property
    def change_end(self):
        """Shot from which the noise is constant again (gradual: end of the ramp)."""
        if self.kind == "gradual":
            return self.change_time + self.ramp_len
        return self.change_start


@dataclass
class GroundTruth:
    """Simulator-only information. Never pass this to a detector or controller."""
    y: np.ndarray             # (n_streams, T) did the baseline correction fail on this shot
    p_t: np.ndarray           # (T,) true flip probability at each shot
    schedule: NoiseSchedule
    realization: np.ndarray   # (n_streams, T, 5*rounds) raw noise draws, for audits


@dataclass
class StreamBatch:
    observable: np.ndarray    # (n_streams, T, 2*rounds+2) syndrome features: all an online method may see
    truth: GroundTruth


def generate_streams(schedule: NoiseSchedule, T: int, q: float, rounds: int, rng, n_streams: int = 1) -> StreamBatch:
    """n_streams independent streams of T shots; shot t uses flip probability schedule.p_at(T)[t]."""
    p_t = schedule.p_at(T)
    n = n_streams
    data = (rng.random((n, T, rounds, 3)) < p_t[None, :, None, None]).astype(np.uint8)
    meas = (rng.random((n, T, rounds, 2)) < q).astype(np.uint8)
    real = np.concatenate([data.reshape(n * T, -1), meas.reshape(n * T, -1)], axis=1)
    X, _, y = rows_from_realization(real, rounds)
    return StreamBatch(observable=X.reshape(n, T, -1),
                       truth=GroundTruth(y=y.reshape(n, T), p_t=p_t, schedule=schedule,
                                         realization=real.reshape(n, T, -1)))