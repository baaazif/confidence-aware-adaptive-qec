"""Exact Bayes-optimal reference decoder, built by enumerating every noise realization.

This is an ORACLE: it knows the noise model (rounds, q, and the set of p values). It is a
reference ceiling for what any decoder could achieve on this data, not a deployable competitor.
With several p values it assumes equal weight, as in the dataset (equal rows per p).
"""
import numpy as np

from caqec.decoders.base import Decoder, check_features, pattern_codes
from caqec.simulation.dataset import DatasetConfig, rows_from_realization


def _enumerate_realizations(rounds: int) -> np.ndarray:
    nbits = 5 * rounds  # 3*rounds data-flip bits + 2*rounds measurement-flip bits
    ids = np.arange(2**nbits, dtype=np.int64)
    return ((ids[:, None] >> np.arange(nbits)) & 1).astype(np.uint8)


class ExactDecoder(Decoder):
    name = "exact"
    provides_confidence = True

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        R = cfg.rounds
        real = _enumerate_realizations(R)
        X, _, y = rows_from_realization(real, R)
        codes = pattern_codes(X)
        data_w = real[:, : 3 * R].sum(axis=1)
        meas_w = real[:, 3 * R:].sum(axis=1)
        size = 2 ** (2 * R + 2)
        joint = np.zeros((size, 2))  # joint[code, label] = P(features=code, y=label)
        for p in cfg.p_values:
            prob = (p**data_w * (1 - p) ** (3 * R - data_w)
                    * cfg.q**meas_w * (1 - cfg.q) ** (2 * R - meas_w))
            for label in (0, 1):
                joint[:, label] += np.bincount(codes[y == label], weights=prob[y == label],
                                               minlength=size) / len(cfg.p_values)
        self.joint = joint
        total = joint.sum(axis=1)
        self._proba = np.where(total > 0, joint[:, 1] / np.where(total > 0, total, 1), 0.5)

    def predict_proba(self, X) -> np.ndarray:
        X = check_features(X, rounds=self.cfg.rounds)
        return self._proba[pattern_codes(X)]

    def expected_failure_rate(self, flip_by_code=None) -> float:
        """Exact failure probability. Default: this decoder; or pass a 0/1 array over all codes."""
        if flip_by_code is None:
            flip_by_code = (self._proba > 0.5).astype(int)
        flip_by_code = np.asarray(flip_by_code, dtype=int)
        wrong = np.where(flip_by_code == 1, self.joint[:, 0], self.joint[:, 1])
        return float(wrong.sum())