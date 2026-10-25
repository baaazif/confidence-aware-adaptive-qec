"""Online change detectors that see ONE number per shot, in time order, and nothing else.

The statistic is the number of detection events in a shot (changes of the syndrome between
consecutive rounds, plus the change into the final syndrome). It rises when the flip probability rises.
No labels, no change times, no future shots: update() receives only the current statistic.
After an alarm the detector re-learns its baseline from the next shots (rebaseline=True)."""
from collections import deque
from math import sqrt

import numpy as np


def defect_counts(X) -> np.ndarray:
    """Detection events per shot from syndrome features (..., 2*rounds+2). Uses observables only."""
    X = np.asarray(X)
    R = (X.shape[-1] - 2) // 2
    m = X[..., : 2 * R].reshape(X.shape[:-1] + (R, 2))
    sf = X[..., -2:]
    prev = np.concatenate([np.zeros(m.shape[:-2] + (1, 2), dtype=m.dtype), m[..., :-1, :]], axis=-2)
    return ((m ^ prev).sum(axis=(-2, -1)) + (sf ^ m[..., -1, :]).sum(axis=-1)).astype(float)


class _Baseline:
    """Welford running mean/variance over the warm-up shots."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def std(self, floor):
        return max(sqrt(self.m2 / (self.n - 1)), floor) if self.n > 1 else floor


class CusumDetector:
    """Two-sided CUSUM on standardized statistics. Baseline = first `warmup` shots after (re)start.
    S+ = max(0, S+ + z - k), S- = max(0, S- - z - k); alarm when either exceeds h."""

    def __init__(self, warmup: int = 500, k: float = 0.5, h: float = 8.0, rebaseline: bool = True,
                 sigma_floor: float = 0.05):
        self.warmup, self.k, self.h, self.rebaseline, self.floor = warmup, k, h, rebaseline, sigma_floor
        self.reset()

    def reset(self):
        self._base, self.mu, self.sigma, self.sp, self.sm = _Baseline(), None, None, 0.0, 0.0

    def update(self, x: float) -> bool:
        if self.mu is None:
            self._base.add(x)
            if self._base.n >= self.warmup:
                self.mu, self.sigma = self._base.mean, self._base.std(self.floor)
            return False
        z = (x - self.mu) / self.sigma
        self.sp = max(0.0, self.sp + z - self.k)
        self.sm = max(0.0, self.sm - z - self.k)
        if self.sp > self.h or self.sm > self.h:
            if self.rebaseline:
                self.reset()
            else:
                self.sp = self.sm = 0.0
            return True
        return False


class WindowDetector:
    """Compare the mean of the latest `w_cur` shots with a frozen reference mean learned from the first
    `w_ref` shots: alarm when |z| > z, z = (mean_cur - mean_ref) / (sd_ref / sqrt(w_cur))."""

    def __init__(self, w_ref: int = 500, w_cur: int = 100, z: float = 5.0, rebaseline: bool = True,
                 sigma_floor: float = 0.05):
        self.w_ref, self.w_cur, self.z, self.rebaseline, self.floor = w_ref, w_cur, z, rebaseline, sigma_floor
        self.reset()

    def reset(self):
        self._base, self.mu, self.sigma = _Baseline(), None, None
        self._win, self._sum = deque(), 0.0

    def update(self, x: float) -> bool:
        if self.mu is None:
            self._base.add(x)
            if self._base.n >= self.w_ref:
                self.mu, self.sigma = self._base.mean, self._base.std(self.floor)
            return False
        self._win.append(x)
        self._sum += x
        if len(self._win) > self.w_cur:
            self._sum -= self._win.popleft()
        if len(self._win) < self.w_cur:
            return False
        zval = (self._sum / self.w_cur - self.mu) / (self.sigma / sqrt(self.w_cur))
        if abs(zval) > self.z:
            if self.rebaseline:
                self.reset()
            else:
                self._win.clear()
                self._sum = 0.0
            return True
        return False


def run_detector(detector, stats) -> list:
    """Feed statistics one at a time, in order. Returns the shot indices at which an alarm fired."""
    update = detector.update
    return [t for t, x in enumerate(np.asarray(stats, dtype=float).tolist()) if update(x)]