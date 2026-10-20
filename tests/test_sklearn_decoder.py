import json
import re
from pathlib import Path

import numpy as np
import pytest

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.selection import candidate_grid, select_best, select_one_se
from caqec.decoders.sklearn_decoder import (FEATURE_SETS, SklearnDecoder, feature_names_for,
                                            featurize)
from caqec.simulation.dataset import DatasetConfig, generate_split

CFG = DatasetConfig()
TREE = {"max_depth": 4, "min_samples_leaf": 20, "class_weight": None}
LOGREG = {"C": 1.0, "class_weight": None}


@pytest.fixture(scope="module")
def data():
    return generate_split(CFG, "train", 3000, np.random.SeedSequence(21))


def test_featurize_known_values_and_shapes():
    X = np.array([[1, 0, 1, 1]], dtype=np.uint8)  # 1 round: m=(1,0), s_final=(1,1)
    assert featurize(X, "raw").tolist() == [[1, 0, 1, 1]]
    assert featurize(X, "events").tolist() == [[1, 0, 0, 1, 1, 1]]  # m, s_final^m, s_final
    pairs = featurize(X, "pairs")
    assert pairs.shape == (1, 4 + 6) and pairs[0, 4] == 0 and pairs[0, -1] == 1  # (c0*c1)=0, (c2*c3)=1
    with pytest.raises(ValueError):
        featurize(X, "nonsense")


def test_feature_names_match_feature_columns(data):
    for fs in FEATURE_SETS:
        assert len(feature_names_for(fs, 3)) == featurize(data["X"][:5], fs).shape[1]
    assert featurize(data["X"][:5], "raw").shape[1] == 8
    assert featurize(data["X"][:5], "events").shape[1] == 10
    assert featurize(data["X"][:5], "pairs").shape[1] == 36


@pytest.mark.parametrize("kind,fs,params", [("tree", "raw", TREE), ("logreg", "pairs", LOGREG)])
def test_decoder_obeys_the_interface_and_physical_failure_rule(data, kind, fs, params):
    d = SklearnDecoder.fit(kind, fs, params, data["X"], data["y"], seed=0)
    proba = d.predict_proba(data["X"])
    assert proba.shape == (len(data["y"]),) and ((proba >= 0) & (proba <= 1)).all()
    assert d.provides_confidence and ((d.confidence(data["X"]) >= 0.5) & (d.confidence(data["X"]) <= 1)).all()
    residual = data["meta_e_final"] ^ d.correction(data["X"])
    weight = residual.sum(axis=1)
    assert set(np.unique(weight)) <= {0, 3}
    assert ((weight == 3) == (d.predict_flip(data["X"]) != data["y"])).all()


def test_training_is_reproducible(data):
    a = SklearnDecoder.fit("tree", "raw", TREE, data["X"], data["y"], seed=5).predict_proba(data["X"])
    b = SklearnDecoder.fit("tree", "raw", TREE, data["X"], data["y"], seed=5).predict_proba(data["X"])
    assert np.array_equal(a, b)
    l1 = SklearnDecoder.fit("logreg", "raw", LOGREG, data["X"], data["y"]).predict_proba(data["X"])
    l2 = SklearnDecoder.fit("logreg", "raw", LOGREG, data["X"], data["y"]).predict_proba(data["X"])
    assert np.array_equal(l1, l2)


def test_save_load_round_trip_and_safety_checks(tmp_path, data):
    d = SklearnDecoder.fit("tree", "raw", TREE, data["X"], data["y"], seed=0)
    d.save(tmp_path / "m")
    loaded = SklearnDecoder.load(tmp_path / "m")
    assert np.array_equal(loaded.predict_proba(data["X"]), d.predict_proba(data["X"]))
    assert loaded.name == d.name and loaded.rounds == d.rounds and loaded.params == d.params
    meta = json.loads((tmp_path / "m.json").read_text())
    assert meta["feature_names"] == feature_names_for("raw", 3) and "sklearn_version" in meta

    wrong_version = dict(meta, sklearn_version="0.0.1")
    (tmp_path / "m.json").write_text(json.dumps(wrong_version))
    with pytest.raises(RuntimeError):
        SklearnDecoder.load(tmp_path / "m")
    assert SklearnDecoder.load(tmp_path / "m", allow_version_mismatch=True) is not None

    (tmp_path / "m.json").write_text(json.dumps(meta))
    blob = tmp_path / "m.joblib"
    blob.write_bytes(blob.read_bytes() + b"x")  # corrupt the model file
    with pytest.raises(ValueError):
        SklearnDecoder.load(tmp_path / "m")


def test_class_weight_balanced_flags_more_positives(data):
    plain = SklearnDecoder.fit("tree", "raw", TREE, data["X"], data["y"], seed=0)
    bal = SklearnDecoder.fit("tree", "raw", {**TREE, "class_weight": "balanced"}, data["X"], data["y"], seed=0)
    assert bal.predict_flip(data["X"]).sum() > plain.predict_flip(data["X"]).sum()


def test_fit_rejects_bad_inputs(data):
    with pytest.raises(ValueError):
        SklearnDecoder.fit("tree", "raw", TREE, data["X"], np.zeros(len(data["y"])))  # one class only
    with pytest.raises(ValueError):
        SklearnDecoder.fit("tree", "raw", TREE, data["X"] * 2, data["y"])             # not binary
    with pytest.raises(ValueError):
        SklearnDecoder.fit("forest", "raw", TREE, data["X"], data["y"])                # unknown kind


def test_raw_logreg_cannot_beat_lookup_signature(data):
    # sanity of the interface: a decoder trained on y can be compared with the lookup baseline
    d = SklearnDecoder.fit("logreg", "raw", LOGREG, data["X"], data["y"])
    assert LookupDecoder().predict_flip(data["X"]).sum() == 0
    assert d.predict_flip(data["X"]).shape == data["y"].shape


def _fake(index, kind, rate, n=10_000):
    return {"index": index, "candidate": {"kind": kind}, "val": {"failure_rate": rate, "n": n}}


def test_selection_rules():
    results = [_fake(0, "tree", 0.0620), _fake(1, "tree", 0.0584), _fake(2, "tree", 0.0565),
               _fake(3, "logreg", 0.0500)]
    assert select_best(results, "tree")["index"] == 2
    # best 0.0565, SE = sqrt(.0565*.9435/10000) ~ 0.00231 -> threshold ~0.0588 -> index 1 is simplest within it
    assert select_one_se(results, "tree")["index"] == 1
    assert select_best(results, "logreg")["index"] == 3        # families are never mixed


def test_candidate_grid_is_well_formed():
    grid = candidate_grid()
    assert len(grid) == 54
    assert {g["kind"] for g in grid} == {"logreg", "tree"}
    assert all(g["feature_set"] in FEATURE_SETS for g in grid)
    assert grid[0]["kind"] == "logreg" and grid[0]["feature_set"] == "raw"


def test_stage2_script_never_opens_the_test_split():
    text = (Path(__file__).resolve().parents[1] / "experiments" / "exp04_stage2_train_select.py").read_text()
    assert "test.npz" not in text
    assert set(re.findall(r'"exp02" / "(\w+)\.npz"', text)) == {"train", "val"}