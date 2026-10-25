import numpy as np
import pytest

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.decoders.sklearn_decoder import featurize
from caqec.detection.online import CusumDetector, WindowDetector, defect_counts, run_detector
from caqec.evaluation.detection_metrics import score_detection, summarize_detection
from caqec.simulation.dataset import DatasetConfig
from caqec.simulation.nonstationary import GroundTruth, NoiseSchedule, StreamBatch, generate_streams

CFG = DatasetConfig()


def test_schedules():
    T = 100
    assert (NoiseSchedule("stationary", 0.05).p_at(T) == 0.05).all()
    a = NoiseSchedule("abrupt", 0.02, 0.1, change_time=40).p_at(T)
    assert (a[:40] == 0.02).all() and (a[40:] == 0.1).all()
    g = NoiseSchedule("gradual", 0.02, 0.10, change_time=20, ramp_len=40).p_at(T)
    assert g[20] == pytest.approx(0.02) and g[40] == pytest.approx(0.06) and g[60] == pytest.approx(0.10) and g[99] == pytest.approx(0.10)
    s = NoiseSchedule("piecewise", 0.01, steps=((30, 0.05), (70, 0.02))).p_at(T)
    assert s[29] == 0.01 and s[30] == 0.05 and s[70] == 0.02
    assert NoiseSchedule("abrupt", 0.02, 0.1, change_time=40).change_start == 40
    assert NoiseSchedule("gradual", 0.02, 0.1, change_time=20, ramp_len=40).change_end == 60
    assert NoiseSchedule("stationary").change_start is None
    for bad in (dict(kind="weird"), dict(p0=1.5), dict(ramp_len=0), dict(kind="piecewise", steps=((5, 0.1), (3, 0.2)))):
        with pytest.raises(ValueError):
            NoiseSchedule(**bad)


def test_streams_are_reproducible_and_follow_the_schedule():
    sch = NoiseSchedule("abrupt", 0.02, 0.10, change_time=500)
    a = generate_streams(sch, 1000, 0.05, 3, np.random.default_rng(1), n_streams=40)
    b = generate_streams(sch, 1000, 0.05, 3, np.random.default_rng(1), n_streams=40)
    assert np.array_equal(a.observable, b.observable) and a.observable.shape == (40, 1000, 8)
    flips = a.truth.realization[..., :9]
    for lo, hi, p in ((0, 500, 0.02), (500, 1000, 0.10)):
        n = flips[:, lo:hi].size
        assert abs(flips[:, lo:hi].mean() - p) < 5 * (p * (1 - p) / n) ** 0.5
    z = generate_streams(NoiseSchedule("stationary", 0.0), 50, 0.0, 3, np.random.default_rng(0))
    assert z.observable.sum() == 0 and z.truth.y.sum() == 0


def test_observable_view_carries_no_ground_truth():
    assert set(StreamBatch.__dataclass_fields__) == {"observable", "truth"}
    assert set(GroundTruth.__dataclass_fields__) == {"y", "p_t", "schedule", "realization"}
    assert "y" not in StreamBatch.__dataclass_fields__
    b = generate_streams(NoiseSchedule(), 20, 0.05, 3, np.random.default_rng(0))
    assert isinstance(b.observable, np.ndarray) and b.observable.dtype == np.uint8


def test_stream_failures_match_exact_analysis():
    from dataclasses import replace
    p = 0.1
    b = generate_streams(NoiseSchedule("stationary", p), 20_000, 0.05, 3, np.random.default_rng(3))
    d = ExactDecoder(replace(CFG, p_values=(p,)))
    expected = d.expected_failure_rate(np.zeros(len(d.joint), dtype=int))
    rate = float((LookupDecoder().predict_flip(b.observable[0]) != b.truth.y[0]).mean())
    assert abs(rate - expected) < 5 * (expected * (1 - expected) / 20_000) ** 0.5


def test_defect_counts_agree_with_event_features():
    X = generate_streams(NoiseSchedule("stationary", 0.1), 300, 0.05, 3, np.random.default_rng(2)).observable[0]
    assert np.array_equal(defect_counts(X), featurize(X, "events")[:, :8].sum(axis=1))
    assert defect_counts(np.zeros((1, 8), dtype=np.uint8))[0] == 0


def _stream(rng, n_before, n_after, mean0=1.0, mean1=2.0):
    return np.concatenate([rng.poisson(mean0, n_before), rng.poisson(mean1, n_after)]).astype(float)


# thresholds chosen for a LOW false-alarm rate: a baseline estimated from a short warm-up inflates
# false alarms well above textbook values, so the experiment script calibrates thresholds by simulation
@pytest.mark.parametrize("make", [lambda: CusumDetector(200, 0.5, 16.0), lambda: WindowDetector(200, 50, 5.5)])
def test_detectors_stay_quiet_then_detect_a_step(make):
    rng = np.random.default_rng(0)
    quiet = sum(len(run_detector(make(), _stream(rng, 3000, 0))) for _ in range(20))
    assert quiet <= 2                                              # almost no false alarms on stationary data
    alarms = run_detector(make(), _stream(rng, 1000, 1000))
    after = [a for a in alarms if a >= 1000]
    assert after and after[0] - 1000 < 300 and not [a for a in alarms if a < 1000]


@pytest.mark.parametrize("make", [lambda: CusumDetector(100, 0.5, 6.0), lambda: WindowDetector(100, 40, 5.0)])
def test_detectors_are_causal(make):
    """Alarms up to time t must not depend on anything after t."""
    rng = np.random.default_rng(1)
    a = _stream(rng, 800, 800)
    b = a.copy()
    b[1000:] = rng.poisson(5.0, len(b) - 1000)                      # change the FUTURE only
    ra, rb = run_detector(make(), a), run_detector(make(), b)
    assert [x for x in ra if x < 1000] == [x for x in rb if x < 1000]


def test_rebaseline_detects_a_second_change_and_constant_input_is_safe():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.poisson(1.0, 1500), rng.poisson(2.5, 1500), rng.poisson(1.0, 1500)]).astype(float)
    alarms = run_detector(CusumDetector(200, 0.5, 8.0), s)
    assert any(1500 <= a < 1900 for a in alarms) and any(a >= 3000 for a in alarms)
    assert run_detector(CusumDetector(50, 0.5, 8.0), np.ones(500)) == []   # zero variance baseline


def test_score_detection_known_cases():
    s = score_detection([100, 3200, 3300], T=6000, warmup=500, change_start=3000)
    assert s == {"false_alarms": 1, "exposure": 2500, "detected": True, "delay": 200}
    assert score_detection([], 6000, 500, 3000)["detected"] is False
    assert score_detection([5000], 6000, 500, 3000, max_delay=1000)["detected"] is False   # too late
    assert score_detection([700, 900], 6000, 500) == {"false_alarms": 2, "exposure": 5500, "detected": None, "delay": None}
    agg = summarize_detection([score_detection([3100], 6000, 500, 3000), score_detection([], 6000, 500, 3000)])
    assert agg["p_detect"] == 0.5 and agg["missed"] == 1 and agg["delay_median"] == 100.0