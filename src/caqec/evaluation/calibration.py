"""Calibration and selective-prediction (reject-by-confidence) measurements. Nothing here fits anything."""
import numpy as np

from caqec.evaluation.stats import wilson_interval


def reliability_table(y, p, n_bins: int = 10) -> list:
    """Bin predictions p = P(y=1) into equal-width bins; compare mean prediction with observed frequency."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, edges[1:-1])  # 0 .. n_bins-1 (p = 1.0 falls in the last bin)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        row = {"bin": b, "low": float(edges[b]), "high": float(edges[b + 1]), "n": n,
               "mean_pred": None, "observed": None, "ci_low": None, "ci_high": None}
        if n:
            k = int(y[m].sum())
            lo, hi = wilson_interval(k, n)
            row.update(mean_pred=float(p[m].mean()), observed=k / n, ci_low=lo, ci_high=hi)
        rows.append(row)
    return rows


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    rows = reliability_table(y, p, n_bins)
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] / total * abs(r["mean_pred"] - r["observed"]) for r in rows if r["n"]))


def truth_gap(p_model, p_true) -> dict:
    """How far a model's probabilities are from the exact posterior, averaged over the rows."""
    d = np.asarray(p_model, dtype=float) - np.asarray(p_true, dtype=float)
    return {"mean_abs": float(np.abs(d).mean()), "rms": float(np.sqrt((d**2).mean()))}


def risk_coverage(y, y_pred, confidence) -> list:
    """Accept shots with confidence >= t for every distinct t; coverage rises as t falls.
    Failure rate is measured among ACCEPTED shots only (rejected shots are discarded)."""
    wrong = np.asarray(y_pred).astype(int) != np.asarray(y).astype(int)
    c = np.asarray(confidence, dtype=float)
    n = len(c)
    points = []
    for t in np.unique(c)[::-1]:
        m = c >= t
        acc, k = int(m.sum()), int(wrong[m].sum())
        lo, hi = wilson_interval(k, acc)
        points.append({"threshold": float(t), "coverage": acc / n, "accepted": acc, "failures": k,
                       "failure_rate": k / acc, "ci_low": lo, "ci_high": hi})
    return points


def at_coverage(points: list, target: float) -> dict:
    """Most selective operating point whose coverage is still >= target (ties can overshoot)."""
    return next(p for p in points if p["coverage"] >= target - 1e-12)