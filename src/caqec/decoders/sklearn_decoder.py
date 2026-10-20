"""scikit-learn decoders behind the common Decoder interface.

Only the feature matrix X is ever used (never meta_ columns, never p). All feature sets are
fixed, parameter-free transforms of X, so no fitted preprocessing can leak between splits.
NOTE: .joblib files are pickles; only load files you created yourself.
"""
import hashlib
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from caqec.decoders.base import Decoder, check_features
from caqec.simulation.dataset import feature_names

FEATURE_SETS = ("raw", "events", "pairs")


def _rounds(X) -> int:
    return (X.shape[1] - 2) // 2


def featurize(X, feature_set: str) -> np.ndarray:
    """raw: the 2R+2 syndrome bits. events: round-to-round changes + final change + s_final.
    pairs: raw plus all pairwise products (lets a linear model see interactions)."""
    X = check_features(X)
    R = _rounds(X)
    if feature_set == "raw":
        return X.astype(np.float64)
    if feature_set == "events":
        m = X[:, : 2 * R].reshape(len(X), R, 2)
        sf = X[:, -2:]
        prev = np.concatenate([np.zeros((len(X), 1, 2), np.uint8), m[:, :-1]], axis=1)
        return np.concatenate([(m ^ prev).reshape(len(X), -1), sf ^ m[:, -1], sf], axis=1).astype(np.float64)
    if feature_set == "pairs":
        pairs = [X[:, i] * X[:, j] for i, j in combinations(range(X.shape[1]), 2)]
        return np.column_stack([X] + pairs).astype(np.float64)
    raise ValueError(f"unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")


def feature_names_for(feature_set: str, rounds: int) -> list:
    raw = feature_names(rounds)
    if feature_set == "raw":
        return raw
    if feature_set == "events":
        ev = [f"event{k}_r{r}" for r in range(1, rounds + 1) for k in (1, 2)]
        return ev + ["event1_final", "event2_final", "s1_final", "s2_final"]
    if feature_set == "pairs":
        return raw + [f"{raw[i]}*{raw[j]}" for i, j in combinations(range(len(raw)), 2)]
    raise ValueError(f"unknown feature_set {feature_set!r}")


def make_estimator(kind: str, params: dict, seed: int = 0):
    if kind == "logreg":
        return LogisticRegression(C=params["C"], class_weight=params.get("class_weight"), max_iter=2000)
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth"),
                                      min_samples_leaf=params.get("min_samples_leaf", 1),
                                      class_weight=params.get("class_weight"), random_state=seed)
    raise ValueError(f"unknown kind {kind!r}")


class SklearnDecoder(Decoder):
    provides_confidence = True

    def __init__(self, estimator, kind, feature_set, params, rounds, meta=None):
        self.estimator, self.kind, self.feature_set = estimator, kind, feature_set
        self.params, self.rounds, self.meta = dict(params), rounds, dict(meta or {})
        self.name = f"{kind}_{feature_set}"

    @classmethod
    def fit(cls, kind, feature_set, params, X, y, seed: int = 0):
        X = check_features(X)
        y = np.asarray(y)
        if not np.isin(y, (0, 1)).all() or len(np.unique(y)) < 2:
            raise ValueError("y must be binary and contain both classes")
        est = make_estimator(kind, params, seed).fit(featurize(X, feature_set), y)
        meta = {"seed": seed, "n_train": int(len(y)), "train_y1_fraction": float(y.mean())}
        return cls(est, kind, feature_set, params, _rounds(X), meta)

    def predict_proba(self, X) -> np.ndarray:
        X = check_features(X, rounds=self.rounds)
        assert list(self.estimator.classes_) == [0, 1]
        return self.estimator.predict_proba(featurize(X, self.feature_set))[:, 1]

    def save(self, path) -> None:
        """Writes <path>.joblib and <path>.json (metadata + checksum)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = path.with_suffix(".joblib")
        joblib.dump(self.estimator, blob)
        meta = {"kind": self.kind, "feature_set": self.feature_set, "params": self.params,
                "rounds": self.rounds, "feature_names": feature_names_for(self.feature_set, self.rounds),
                "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), **self.meta}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path, allow_version_mismatch: bool = False):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        blob = path.with_suffix(".joblib")
        if hashlib.sha256(blob.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError("model file does not match its recorded checksum")
        if meta["sklearn_version"] != sklearn.__version__ and not allow_version_mismatch:
            raise RuntimeError(f"model saved with scikit-learn {meta['sklearn_version']}, "
                               f"installed {sklearn.__version__}")
        extra = {k: meta[k] for k in ("seed", "n_train", "train_y1_fraction") if k in meta}
        return cls(joblib.load(blob), meta["kind"], meta["feature_set"], meta["params"],
                   meta["rounds"], extra)