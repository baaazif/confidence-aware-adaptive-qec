from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from caqec.plotting.nonstationary_plots import plot_detection_summary, plot_failure_timeseries

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "exp10_stage8_nonstationary_detector.py"


def test_script_opens_no_dataset_and_detectors_see_only_observables():
    text = SCRIPT.read_text()
    for forbidden in ("test.npz", "train.npz", "val.npz", "load_split(", "SklearnDecoder.fit", "candidate_grid"):
        assert forbidden not in text, forbidden
    for line in text.splitlines():
        if "run_detector(" in line and "def " not in line:
            assert "truth" not in line and "y[" not in line and "wrong" not in line, line
        if "stats = defect_counts(" in line:
            assert ".observable" in line or "(X)" in line, line


def test_plots_write_files(tmp_path):
    rate = np.full(24, 0.05)
    panels = [{"title": "t", "curves": {"a": (rate, rate * 0.9, rate * 1.1), "b": (rate * 2, rate * 1.8, rate * 2.2)},
               "change_start": 3000, "change_end": 4500, "median_alarm": {"cusum": 3100.0, "window": None}}] * 5
    plot_failure_timeseries(panels, 250, tmp_path / "a.png")
    summ = {"delay_median": 100.0, "delay_p90": 200.0, "p_detect": 1.0, "p_detect_ci": [0.96, 0.9999999999999999]}
    plot_detection_summary([{"label": "x", "cusum": summ, "window": summ}], tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 2000 and (tmp_path / "b.png").stat().st_size > 2000