"""Candidate grid and validation-based model selection (the test split is never involved)."""
from math import sqrt

from caqec.decoders.sklearn_decoder import SklearnDecoder
from caqec.evaluation.decoder_metrics import evaluate


def candidate_grid() -> list:
    """Ordered simple -> complex within each family (the order is used by the 1-SE rule)."""
    grid = []
    for fs in ("raw", "events", "pairs"):
        for C in (0.1, 1.0, 10.0):
            for cw in (None, "balanced"):
                grid.append({"kind": "logreg", "feature_set": fs, "params": {"C": C, "class_weight": cw}})
    for depth in (3, 4, 5, 6, 8, None):
        for leaf in (50, 20, 1):
            for cw in (None, "balanced"):
                grid.append({"kind": "tree", "feature_set": "raw",
                             "params": {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": cw}})
    return grid


def fit_and_score(grid, X_train, y_train, X_val, y_val, seed: int = 0) -> list:
    """Fit every candidate on TRAIN only and score it on VALIDATION."""
    results = []
    for i, cand in enumerate(grid):
        dec = SklearnDecoder.fit(cand["kind"], cand["feature_set"], cand["params"], X_train, y_train, seed)
        proba = dec.predict_proba(X_val)
        results.append({"index": i, "candidate": cand, "decoder": dec,
                        "val": evaluate(y_val, (proba > 0.5).astype(int), proba)})
    return results


def select_best(results, kind):
    pool = [r for r in results if r["candidate"]["kind"] == kind]
    return min(pool, key=lambda r: (r["val"]["failure_rate"], r["index"]))


def select_one_se(results, kind):
    """Simplest candidate whose validation failure rate is within one standard error of the best."""
    pool = [r for r in results if r["candidate"]["kind"] == kind]
    best = min(r["val"]["failure_rate"] for r in pool)
    se = sqrt(best * (1 - best) / pool[0]["val"]["n"])
    return next(r for r in pool if r["val"]["failure_rate"] <= best + se)