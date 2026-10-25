"""Design-time model knowledge available to a controller: the noise-model FAMILY (data flips p, measurement
flips q, fixed code) and offline decoder tables. It is NOT information about the live stream: the controller
does not know p_t, q_t or when anything changes. Everything is exact, via a count tensor over noise patterns:
P(pattern, label | p, q) = sum_{k,m} count[pattern, label, k, m] * p^k (1-p)^(3R-k) * q^m (1-q)^(2R-m)."""
import numpy as np

from caqec.decoders.base import pattern_codes
from caqec.decoders.exact import _enumerate_realizations
from caqec.detection.online import defect_counts
from caqec.simulation.dataset import rows_from_realization

P_STEP, Q_STEP = 0.0025, 0.005
P_GRID = np.arange(0.0, 0.4 + 1e-9, P_STEP)
Q_GRID = np.arange(0.0, 0.3 + 1e-9, Q_STEP)


class NoiseKnowledge:
    def __init__(self, rounds: int, static_tables: dict, monitor_proba):
        R = self.R = rounds
        real = _enumerate_realizations(R)
        X, _, y = rows_from_realization(real, R)
        codes = pattern_codes(X)
        dw = real[:, : 3 * R].sum(axis=1).astype(np.int64)
        mw = real[:, 3 * R:].sum(axis=1).astype(np.int64)
        y = y.astype(np.int64)
        self.n_codes = 2 ** (2 * R + 2)
        self.nk, self.nm = 3 * R + 1, 2 * R + 1
        idx = ((codes * 2 + y) * self.nk + dw) * self.nm + mw
        self.cnt = np.bincount(idx, minlength=self.n_codes * 2 * self.nk * self.nm).reshape(
            self.n_codes, 2, self.nk, self.nm).astype(float)
        all_x = ((np.arange(self.n_codes)[:, None] >> np.arange(2 * R + 2)) & 1).astype(np.uint8)
        self.events = defect_counts(all_x)
        self.static = {k: np.asarray(v, dtype=np.uint8) for k, v in static_tables.items()}
        mp = np.asarray(monitor_proba, dtype=float)
        self.selfp = np.minimum(mp, 1 - mp)          # the monitor decoder's own estimate of its failure probability
        self._bayes = {}
        self._E = np.empty((len(Q_GRID), len(P_GRID)))
        self._S = np.empty_like(self._E)
        for i, q in enumerate(Q_GRID):
            for j, p in enumerate(P_GRID):
                tot = self.joint(p, q).sum(axis=1)
                self._E[i, j], self._S[i, j] = tot @ self.events, tot @ self.selfp

    def joint(self, p: float, q: float) -> np.ndarray:
        k, m = np.arange(self.nk), np.arange(self.nm)
        a = p**k * (1 - p) ** (3 * self.R - k)
        b = q**m * (1 - q) ** (2 * self.R - m)
        return np.einsum("clkm,k,m->cl", self.cnt, a, b)

    @staticmethod
    def quant(p: float, q: float):
        return (round(min(max(p, 0.0), 0.4) / P_STEP) * P_STEP, round(min(max(q, 0.0), 0.3) / Q_STEP) * Q_STEP)

    def table(self, spec) -> np.ndarray:
        """Flip table over all patterns. spec: a static decoder name, or ('bayes', p, q) = exact decoder for (p, q)."""
        if isinstance(spec, str):
            return self.static[spec]
        if spec not in self._bayes:
            j = self.joint(spec[1], spec[2])
            self._bayes[spec] = (j[:, 1] > j[:, 0]).astype(np.uint8)
        return self._bayes[spec]

    def failure(self, spec, p: float, q: float) -> float:
        j = self.joint(p, q)
        return float(np.where(self.table(spec) == 1, j[:, 0], j[:, 1]).sum())

    def mean_events(self, p, q):
        j = self.joint(p, q).sum(axis=1)
        return float(j @ self.events)

    def mean_selfp(self, p, q):
        j = self.joint(p, q).sum(axis=1)
        return float(j @ self.selfp)

    def _invert(self, grid, target: float, q: float) -> float:
        q = min(max(q, 0.0), Q_GRID[-1])
        i = min(int(q / Q_STEP), len(Q_GRID) - 2)
        w = (q - Q_GRID[i]) / Q_STEP
        curve = (1 - w) * grid[i] + w * grid[i + 1]
        return float(np.interp(target, curve, P_GRID))        # curves rise with p, so this inverts them

    def estimate(self, n: int, ev_mean, ev_sd, mm_mean, sp_mean, sp_sd, z: float = 2.0, q_fixed=None) -> dict:
        """Noise estimates from OBSERVABLE window statistics only (event count, last-round mismatch, monitor self-failure)."""
        if q_fixed is None:
            q_hat = min(max(mm_mean / 2, 0.0), 0.3)           # E[mismatch] = 2q exactly, whatever p is
            se_q = (max(q_hat * (1 - q_hat), 0.01) / (2 * n)) ** 0.5
        else:
            q_hat, se_q = q_fixed, 0.0
        q_lo, q_hi = max(q_hat - z * se_q, 0.0), min(q_hat + z * se_q, 0.3)
        se_ev, se_sp = ev_sd / n**0.5, sp_sd / n**0.5
        out = {"q_hat": q_hat, "q_lo": q_lo, "q_hi": q_hi}
        for tag, grid, mean, se in (("ev", self._E, ev_mean, se_ev), ("cf", self._S, sp_mean, se_sp)):
            out[f"p_hat_{tag}"] = self._invert(grid, mean, q_hat)
            out[f"p_lo_{tag}"] = self._invert(grid, mean - z * se, q_hi)    # more q explains the signal -> smaller p
            out[f"p_hi_{tag}"] = self._invert(grid, mean + z * se, q_lo)
        return out