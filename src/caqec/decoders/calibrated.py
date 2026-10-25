"""Post-hoc calibration of a decoder's probabilities (fitted on VALIDATION data only).

The base decoder is never changed. The calibrator maps its P(y=1) to a better-calibrated P(y=1).
sigmoid : Platt scaling, a logistic curve on the logit of the base probability (keeps the ranking).
isotonic: a monotone step function (can merge nearby probabilities into ties).
NOTE: predict_flip thresholds the calibrated probability at 0.5, so decisions can change slightly.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from caqec.decoders.base import Decoder, check_features

METHODS = ("sigmoid", "isotonic")
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class CalibratedDecoder(Decoder):
    provides_confidence = True

    def __init__(self, base: Decoder, method: str, calibrator, n_fit: int):
        self.base, self.method, self.calibrator, self.n_fit = base, method, calibrator, n_fit
        self.name = f"{base.name}+{method}"

    @classmethod
    def fit(cls, base: Decoder, X_cal, y_cal, method: str):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not base.provides_confidence:
            raise ValueError(f"{base.name} has no probabilities to calibrate")
        y_cal = np.asarray(y_cal).astype(int)
        scores = base.predict_proba(check_features(X_cal))
        if method == "isotonic":
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y_cal)
        else:
            cal = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores).reshape(-1, 1), y_cal)
        return cls(base, method, cal, int(len(y_cal)))

    def predict_proba(self, X) -> np.ndarray:
        scores = self.base.predict_proba(X)
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(scores), 0.0, 1.0)
        return self.calibrator.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]