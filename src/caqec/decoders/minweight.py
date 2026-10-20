"""History-aware classical decoder: minimum-weight explanation of the whole syndrome history.

For each possible input pattern it finds the cheapest noise realization (data flips and
measurement flips, weighted by log-odds from NOMINAL p and q) that explains the pattern, then
decodes as if that realization happened. This is the standard 'matching-style' classical decoder,
done here by exhaustive search because the code is tiny. It is configured once with nominal
noise levels and is NOT told the true p of each shot.

Rules: if several equally cheap realizations disagree on the label, predict 0 (no extra flip,
i.e. fall back to the plain lookup). Unreachable patterns also get 0.
With p_nom == q_nom both fault types cost the same, so it simply minimizes the NUMBER of faults.
"""
from math import log

import numpy as np

from caqec.decoders.base import Decoder, check_features, pattern_codes
from caqec.decoders.exact import _enumerate_realizations  # internal helper shared inside the package
from caqec.simulation.dataset import rows_from_realization


class MinWeightDecoder(Decoder):
    provides_confidence = False  # a hard decision; it has no probability

    def __init__(self, rounds: int = 3, p_nom: float = 0.05, q_nom: float = 0.05):
        if not (0 < p_nom < 0.5 and 0 < q_nom < 0.5):
            raise ValueError("nominal p and q must be in (0, 0.5)")
        self.rounds, self.p_nom, self.q_nom = rounds, p_nom, q_nom
        self.name = f"minweight(p={p_nom},q={q_nom})"
        R = rounds
        real = _enumerate_realizations(R)
        X, _, y = rows_from_realization(real, R)
        codes = pattern_codes(X)
        weight = (log((1 - p_nom) / p_nom) * real[:, : 3 * R].sum(axis=1)
                  + log((1 - q_nom) / q_nom) * real[:, 3 * R:].sum(axis=1))
        size = 2 ** (2 * R + 2)
        best = np.full(size, np.inf)
        np.minimum.at(best, codes, weight)
        cheapest = weight <= best[codes] + 1e-9
        ones = np.bincount(codes[cheapest & (y == 1)], minlength=size)
        zeros = np.bincount(codes[cheapest & (y == 0)], minlength=size)
        self._flip = ((ones > 0) & (zeros == 0)).astype(np.uint8)  # flip only if ALL cheapest agree on 1

    def predict_proba(self, X) -> np.ndarray:
        X = check_features(X, rounds=self.rounds)
        return self._flip[pattern_codes(X)].astype(float)