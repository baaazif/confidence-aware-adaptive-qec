"""Common decoder interface (input: dataset feature matrix X; output: residual-flip decision).

Convention (matches docs/dataset_spec.md):
  y = 1 means "the baseline correction for s_final needs an extra logical flip (XXX)".
  A decoder predicts y from X only. Its full correction is
      reference_correction(s_final) XOR (y_hat * 111)
  and the shot is a logical failure exactly when y_hat != y.
"""
from abc import ABC, abstractmethod

import numpy as np

from caqec.simulation.dataset import reference_correction


def check_features(X, rounds=None) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 4 or X.shape[1] % 2 != 0:
        raise ValueError(f"X must be 2-D with an even number (>= 4) of columns, got {X.shape}")
    if not np.isin(X, (0, 1)).all():
        raise ValueError("X must be binary")
    if rounds is not None and X.shape[1] != 2 * rounds + 2:
        raise ValueError(f"expected {2 * rounds + 2} columns for {rounds} rounds, got {X.shape[1]}")
    return X.astype(np.uint8)


def pattern_codes(X) -> np.ndarray:
    """Binary feature rows -> integer ids (one per distinct input pattern)."""
    X = np.asarray(X, dtype=np.int64)
    return X @ (1 << np.arange(X.shape[1], dtype=np.int64))


class Decoder(ABC):
    name = "decoder"
    provides_confidence = False  # True only if predict_proba is a meaningful probability

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """P(y = 1 | X), shape (n,). Hard decoders return 0/1 and set provides_confidence=False."""

    def predict_flip(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(np.uint8)  # ties -> 0 (= baseline)

    def confidence(self, X) -> np.ndarray:
        """Probability the decoder assigns to its own decision (hook for the adaptive controller)."""
        if not self.provides_confidence:
            raise NotImplementedError(f"{self.name} does not provide calibrated confidence")
        p = self.predict_proba(X)
        return np.maximum(p, 1 - p)

    def correction(self, X) -> np.ndarray:
        """Full (n, 3) correction to apply to the data qubits."""
        X = check_features(X)
        flip = self.predict_flip(X).astype(np.uint8)
        return reference_correction(X[:, -2:]) ^ (flip[:, None] * np.ones(3, dtype=np.uint8))