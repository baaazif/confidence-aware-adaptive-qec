from math import sqrt

import numpy as np
from scipy.stats import binomtest

from caqec.evaluation.stats import wilson_interval


def _ratio(num, den):
    return None if den == 0 else num / den


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    """Metrics for one decoder. Logical failure <=> y_pred != y_true, so accuracy = 1 - failure."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = n - tp - fp - fn
    k = fp + fn
    lo, hi = wilson_interval(k, n)
    recall, fpr = _ratio(tp, tp + fn), _ratio(fp, fp + tn)
    out = {"n": n, "failures": k, "failure_rate": k / n, "ci_low": lo, "ci_high": hi,
           "accuracy": 1 - k / n, "recall_y1": recall, "precision_y1": _ratio(tp, tp + fp),
           "false_positive_rate": fpr,
           "balanced_accuracy": None if recall is None or fpr is None else (recall + 1 - fpr) / 2,
           "n_y1": tp + fn, "predicted_y1": tp + fp,
           "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}
    if y_proba is not None:
        p = np.clip(np.asarray(y_proba, dtype=float), 1e-12, 1 - 1e-12)
        out["brier"] = float(np.mean((p - y_true) ** 2))
        out["log_loss"] = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return out


def evaluate_by_group(y_true, y_pred, groups, y_proba=None) -> dict:
    """Same metrics per distinct value in `groups` (e.g. the p column). Evaluation only."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    res = {}
    for g in np.unique(groups):
        m = groups == g
        res[float(g)] = evaluate(y_true[m], y_pred[m], None if y_proba is None else np.asarray(y_proba)[m])
    return res


def paired_difference(y_true, pred_a, pred_b, z: float = 1.96) -> dict:
    """failure(A) - failure(B) on the SAME rows, with a Wald CI and an exact McNemar p-value."""
    y_true = np.asarray(y_true).astype(int)
    wrong_a = np.asarray(pred_a).astype(int) != y_true
    wrong_b = np.asarray(pred_b).astype(int) != y_true
    n = len(y_true)
    b = int((wrong_a & ~wrong_b).sum())  # A wrong, B right
    c = int((~wrong_a & wrong_b).sum())  # A right, B wrong
    diff = (b - c) / n
    se = sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    pval = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"diff": diff, "ci_low": diff - z * se, "ci_high": diff + z * se,
            "a_wrong_b_right": b, "a_right_b_wrong": c, "mcnemar_p": pval, "n": n}