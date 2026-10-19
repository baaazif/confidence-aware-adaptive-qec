import numpy as np
import pytest

from caqec.decoders.base import Decoder, check_features, pattern_codes
from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.evaluation.dataset_checks import p_effective
from caqec.evaluation.decoder_metrics import evaluate, evaluate_by_group, paired_difference
from caqec.simulation.dataset import DatasetConfig, generate_split, reference_correction

CFG = DatasetConfig()
R1 = DatasetConfig(rounds=1, q=0.0, p_values=(0.1,))


@pytest.fixture(scope="module")
def exact():
    return ExactDecoder(CFG)


def small_split(cfg=CFG, n=2000, seed=3):
    return generate_split(cfg, "train", n, np.random.SeedSequence(seed))


class AlwaysFlip(Decoder):
    name = "always_flip"

    def predict_proba(self, X):
        return np.ones(len(check_features(X)))


def test_lookup_predicts_no_flip_and_has_no_confidence():
    d = LookupDecoder()
    X = np.array([[0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    assert d.predict_flip(X).tolist() == [0, 0]
    assert d.correction(X).tolist() == [[1, 0, 0], [0, 0, 0]]  # syndrome (1,0) -> flip qubit 0
    assert not d.provides_confidence
    with pytest.raises(NotImplementedError):
        d.confidence(X)


def test_flip_decision_complements_the_reference_correction():
    X = small_split(n=200)["X"]
    base = reference_correction(X[:, -2:])
    assert (AlwaysFlip().correction(X) == (base ^ 1)).all()


@pytest.mark.parametrize("make", [lambda: LookupDecoder(), lambda: ExactDecoder(CFG), lambda: AlwaysFlip()])
def test_failure_equals_label_mismatch_physically(make):
    s = small_split()
    d = make()
    residual = s["meta_e_final"] ^ d.correction(s["X"])      # error left after applying the correction
    weight = residual.sum(axis=1)
    assert set(np.unique(weight)) <= {0, 3}                  # always a codeword
    assert ((weight == 3) == (d.predict_flip(s["X"]) != s["y"])).all()


def test_exact_decoder_reduces_to_lookup_in_the_baseline_noise_model():
    cfg, p = R1, 0.1
    ex = ExactDecoder(cfg)
    s = small_split(cfg, n=20000)
    X = s["X"]
    nontrivial = X[:, -2:].any(axis=1)
    proba = ex.predict_proba(X)
    assert np.allclose(proba[nontrivial], p, atol=1e-12)
    assert np.allclose(proba[~nontrivial], p**3 / ((1 - p) ** 3 + p**3), atol=1e-12)
    assert ex.predict_flip(X).sum() == 0                      # never flips: identical to lookup
    assert ex.expected_failure_rate() == pytest.approx(3 * p**2 - 2 * p**3, abs=1e-12)


def test_exact_decoder_probabilities_are_consistent(exact):
    assert exact.joint.sum() == pytest.approx(1.0, abs=1e-12)
    assert ((exact._proba >= 0) & (exact._proba <= 1)).all()
    zeros = np.zeros(len(exact.joint), dtype=int)
    analytic_lookup = np.mean([3 * p_effective(p, 3) ** 2 - 2 * p_effective(p, 3) ** 3 for p in CFG.p_values])
    assert exact.expected_failure_rate(zeros) == pytest.approx(analytic_lookup, abs=1e-12)
    assert exact.expected_failure_rate() <= exact.expected_failure_rate(zeros)


def test_enumeration_agrees_with_monte_carlo_sampling(exact):
    s = generate_split(CFG, "train", 20_000, np.random.SeedSequence(11))
    rate = float((exact.predict_flip(s["X"]) != s["y"]).mean())
    expected = exact.expected_failure_rate()
    sigma = (expected * (1 - expected) / len(s["y"])) ** 0.5
    assert abs(rate - expected) < 5 * sigma


def test_exact_decoder_is_deterministic():
    assert np.array_equal(ExactDecoder(CFG).joint, ExactDecoder(CFG).joint)


def test_input_validation(exact):
    with pytest.raises(ValueError):
        check_features(np.array([[0, 2, 0, 0]]))
    with pytest.raises(ValueError):
        exact.predict_proba(np.zeros((3, 4), dtype=np.uint8))  # wrong number of rounds
    assert pattern_codes(np.array([[1, 0, 1, 0]])).tolist() == [5]


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])
    m = evaluate(y, pred)
    assert m["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert m["failure_rate"] == pytest.approx(2 / 6) and m["accuracy"] == pytest.approx(4 / 6)
    assert m["recall_y1"] == pytest.approx(2 / 3) and m["precision_y1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert evaluate([0, 0], [0, 0])["recall_y1"] is None     # no positives: undefined, not zero
    groups = evaluate_by_group(y, pred, np.array([1, 1, 1, 2, 2, 2]))
    assert groups[1.0]["failures"] == 1 and groups[2.0]["failures"] == 1


def test_paired_difference_known_values():
    y = np.array([0, 0, 1, 1])
    a = np.array([0, 1, 0, 1])   # wrong at rows 1, 2
    b = np.array([0, 0, 1, 0])   # wrong at row 3
    d = paired_difference(y, a, b)
    assert d["a_wrong_b_right"] == 2 and d["a_right_b_wrong"] == 1
    assert d["diff"] == pytest.approx(0.25)
    assert paired_difference(y, a, a)["diff"] == 0 and paired_difference(y, a, a)["mcnemar_p"] == 1.0