"""Noise-shift scenarios for generalization tests (fresh data, separate seed family)."""
import numpy as np

TRAINED_P = (0.02, 0.05, 0.10)
BASE_Q = 0.05
P_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
Q_GRID = (0.0, 0.02, 0.10, 0.20)
Q_SWEEP_P = 0.05
SHIFT_SEED = 20261020  # different entropy from the dataset's master seed (20261019)


def region_for_p(p: float, trained=TRAINED_P) -> str:
    if any(np.isclose(p, t) for t in trained):
        return "trained"
    return "interpolation" if min(trained) < p < max(trained) else "extrapolation"


def build_scenarios() -> list:
    scenarios = [{"name": f"p={p}", "axis": "p", "p": p, "q": BASE_Q, "region": region_for_p(p)}
                 for p in P_GRID]
    scenarios += [{"name": f"q={q}", "axis": "q", "p": Q_SWEEP_P, "q": q, "region": "q-shift"}
                  for q in Q_GRID]
    return scenarios


def scenario_seeds(n: int) -> list:
    return np.random.SeedSequence(SHIFT_SEED).spawn(n)