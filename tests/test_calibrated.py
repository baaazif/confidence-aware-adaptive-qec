from pathlib import Path
import re

import numpy as np
import pytest

from caqec.decoders.base import Decoder, check_features, pattern_codes
from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.calibrated import CalibratedDecoder
from caqec.decoders.sklearn_decoder import SklearnDecoder
from caqec.evaluation.calibration import expected_calibration_error
from caqec.simulation.dataset import DatasetConfig, generate_split

CFG = DatasetConfig()


class OverconfidentOnFirstBit(Decoder):
    """Stub: claims 2% / 98% depending on bit 0, while the truth is 10% / 60%."""
    name = "stub"
    provides_confidence = True

    def predict_proba(self, X):
        return 0.02 + 0.96 * check_features(X)[:, 0]


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(40_000, 8)).astype(np.uint8)
    y = (rng.random(40_000) < 0.1 + 0.5 * X[:, 0]).astype(int)
    return X[:20_000], y[:20_000], X[20_000:], y[20_000:]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_repairs_an_overconfident_decoder_on_held_out_data(synthetic, method):
    Xc, yc, Xt, yt = synthetic
    base = OverconfidentOnFirstBit()
    raw = expected_calibration_error(yt, base.predict_proba(Xt))
    cal = CalibratedDecoder.fit(base, Xc, yc, method)
    assert raw > 0.2 and expected_calibration_error(yt, cal.predict_proba(Xt)) < 0.03


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_is_monotone_and_bounded(synthetic, method):
    Xc, yc, Xt, _ = synthetic
    cal = CalibratedDecoder.fit(OverconfidentOnFirstBit(), Xc, yc, method)
    grid = np.linspace(0, 1, 201)
    if method == "isotonic":
        out = cal.calibrator.predict(grid)
    else:
        from caqec.decoders.calibrated import _logit
        out = cal.calibrator.predict_proba(_logit(grid).reshape(-1, 1))[:, 1]
    assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1


def test_calibrating_a_real_model_keeps_the_interface_and_physical_rule():
    s = generate_split(CFG, "train", 4000, np.random.SeedSequence(4))
    base = SklearnDecoder.fit("tree", "raw", {"max_depth": 4, "min_samples_leaf": 20, "class_weight": None},
                              s["X"], s["y"], seed=0)
    cal = CalibratedDecoder.fit(base, s["X"][:2000], s["y"][:2000], "isotonic")
    assert cal.name == "tree_raw+isotonic" and cal.provides_confidence
    p = cal.predict_proba(s["X"])
    assert ((p >= 0) & (p <= 1)).all()
    weight = (s["meta_e_final"] ^ cal.correction(s["X"])).sum(axis=1)
    assert set(np.unique(weight)) <= {0, 3}
    assert ((weight == 3) == (cal.predict_flip(s["X"]) != s["y"])).all()


def test_calibration_rejects_bad_requests(synthetic):
    Xc, yc, _, _ = synthetic
    with pytest.raises(ValueError):
        CalibratedDecoder.fit(OverconfidentOnFirstBit(), Xc, yc, "platt")
    with pytest.raises(ValueError):
        CalibratedDecoder.fit(LookupDecoder(), Xc, yc, "sigmoid")      # no probabilities to calibrate


def test_stage7_script_fits_calibrators_on_validation_only():
    text = (Path(__file__).resolve().parents[1] / "experiments" / "exp09_stage7_calibration_fit.py").read_text()
    for forbidden in ("test.npz", "train.npz", "SklearnDecoder", "candidate_grid"):
        assert forbidden not in text, forbidden
    fit_lines = [ln for ln in text.splitlines() if "CalibratedDecoder.fit(" in ln]
    assert fit_lines and all('val["X"]' in ln and 'val["y"]' in ln for ln in fit_lines)