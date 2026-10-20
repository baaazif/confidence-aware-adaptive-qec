from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from caqec.decoders.exact import ExactDecoder
from caqec.evaluation.calibration import (at_coverage, expected_calibration_error, reliability_table,
                                          risk_coverage, truth_gap)
from caqec.plotting.calibration_plots import plot_reliability, plot_risk_coverage
from caqec.simulation.dataset import DatasetConfig, generate_split


def test_reliability_table_counts_and_edges():
    p = np.array([0.0, 0.05, 0.35, 0.99, 1.0])
    y = np.array([0, 0, 1, 1, 1])
    rows = reliability_table(y, p, n_bins=10)
    assert sum(r["n"] for r in rows) == 5
    assert rows[0]["n"] == 2 and rows[3]["n"] == 1 and rows[9]["n"] == 2   # p = 1.0 lands in the last bin
    assert rows[1]["mean_pred"] is None                                    # empty bins stay empty


def test_ece_known_values():
    y = np.array([1, 1, 1] + [0] * 7)
    assert expected_calibration_error(y, np.full(10, 0.3)) == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(np.zeros(10, dtype=int), np.full(10, 0.8)) == pytest.approx(0.8)


def test_truth_gap_known_values():
    assert truth_gap([0.2, 0.4], [0.2, 0.4]) == {"mean_abs": 0.0, "rms": 0.0}
    g = truth_gap([0.1, 0.5], [0.2, 0.3])
    assert g["mean_abs"] == pytest.approx(0.15) and g["rms"] == pytest.approx(np.sqrt((0.01 + 0.04) / 2))


def test_risk_coverage_small_example():
    y = np.array([0, 0, 0, 1])
    pred = np.array([0, 0, 0, 0])
    conf = np.array([0.9, 0.9, 0.9, 0.6])
    pts = risk_coverage(y, pred, conf)
    assert [p["coverage"] for p in pts] == [0.75, 1.0]
    assert [p["failures"] for p in pts] == [0, 1] and pts[1]["failure_rate"] == pytest.approx(0.25)
    assert at_coverage(pts, 0.7)["coverage"] == 0.75 and at_coverage(pts, 1.0)["coverage"] == 1.0


def test_risk_coverage_is_well_formed_and_oracle_selection_helps():
    cfg = DatasetConfig()
    d = ExactDecoder(cfg)
    s = generate_split(cfg, "train", 15_000, np.random.SeedSequence(8))
    pts = risk_coverage(s["y"], d.predict_flip(s["X"]), d.confidence(s["X"]))
    cov = [p["coverage"] for p in pts]
    assert cov == sorted(cov) and cov[-1] == 1.0
    full = pts[-1]["failure_rate"]
    assert at_coverage(pts, 0.9)["failure_rate"] < full                # rejecting the doubtful 10% helps


def test_plots_write_files(tmp_path):
    rows = reliability_table(np.array([0, 1, 0, 1]), np.array([0.1, 0.6, 0.3, 0.9]))
    plot_reliability({"a": rows}, tmp_path / "r.png", "t")
    pts = risk_coverage(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0.9, 0.8, 0.6]))
    plot_risk_coverage({"a": pts}, tmp_path / "c.png", "t", reference={"lookup": 0.07})
    for f in ("r.png", "c.png"):
        assert (tmp_path / f).stat().st_size > 2000


def test_stage6_script_opens_no_test_split_and_fits_nothing():
    text = (Path(__file__).resolve().parents[1] / "experiments" / "exp08_stage6_calibration_risk.py").read_text()
    for forbidden in ("test.npz", "train.npz", "fit(", "candidate_grid", "CalibratedClassifier", "isotonic"):
        assert forbidden not in text, forbidden