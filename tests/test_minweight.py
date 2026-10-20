from itertools import product

import numpy as np
import pytest

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.minweight import MinWeightDecoder
from caqec.simulation.dataset import DatasetConfig, generate_split

CFG = DatasetConfig()
HISTORY = np.array([[1, 1, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)  # q1 flips, then q2 flips, rest persists


def test_interface_and_hard_decisions():
    d = MinWeightDecoder()
    X = generate_split(CFG, "train", 500, np.random.SeedSequence(1))["X"]
    assert not d.provides_confidence
    assert set(np.unique(d.predict_proba(X))) <= {0.0, 1.0}
    assert set(np.unique(d.predict_flip(X))) <= {0, 1}
    with pytest.raises(NotImplementedError):
        d.confidence(X)
    with pytest.raises(ValueError):
        d.predict_proba(np.zeros((2, 4), dtype=np.uint8))          # wrong number of rounds
    with pytest.raises(ValueError):
        MinWeightDecoder(p_nom=0.7)


def test_failure_equals_label_mismatch_physically():
    s = generate_split(CFG, "train", 2000, np.random.SeedSequence(3))
    d = MinWeightDecoder()
    weight = (s["meta_e_final"] ^ d.correction(s["X"])).sum(axis=1)
    assert set(np.unique(weight)) <= {0, 3}
    assert ((weight == 3) == (d.predict_flip(s["X"]) != s["y"])).all()


def test_single_round_consistent_patterns_match_the_lookup():
    d, lookup = MinWeightDecoder(rounds=1), LookupDecoder()
    for s1, s2 in product((0, 1), repeat=2):
        X = np.array([[s1, s2, s1, s2]], dtype=np.uint8)           # measurement agrees with final syndrome
        assert d.predict_flip(X)[0] == lookup.predict_flip(X)[0] == 0


def test_persistent_single_flip_is_not_overridden():
    X = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)       # q0 flipped in round 1, stays
    assert MinWeightDecoder().predict_flip(X)[0] == 0


def test_tie_falls_back_to_no_flip_and_weights_can_break_it():
    # two faults explain HISTORY either way: (2 data flips -> y=1) or (1 data + 1 measurement flip -> y=0)
    assert MinWeightDecoder(p_nom=0.05, q_nom=0.05).predict_flip(HISTORY)[0] == 0    # tie -> no flip
    assert MinWeightDecoder(p_nom=0.10, q_nom=0.05).predict_flip(HISTORY)[0] == 1    # data faults cheaper
    assert MinWeightDecoder(p_nom=0.02, q_nom=0.05).predict_flip(HISTORY)[0] == 0    # data faults dearer


def test_deterministic_table():
    assert np.array_equal(MinWeightDecoder()._flip, MinWeightDecoder()._flip)


def test_stage5_script_never_opens_the_test_split_or_trains():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "experiments" / "exp07_stage5_minweight_baseline.py").read_text()
    for forbidden in ("test.npz", "train.npz", "fit(", "candidate_grid"):
        assert forbidden not in text, forbidden