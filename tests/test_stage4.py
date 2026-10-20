from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.sklearn_decoder import SklearnDecoder
from caqec.evaluation.frozen import load_frozen_models
from caqec.evaluation.shift_eval import run_scenario
from caqec.plotting.shift_plots import plot_series
from caqec.simulation.dataset import DatasetConfig, generate_split
from caqec.simulation.shift import SHIFT_SEED, TRAINED_P, build_scenarios, region_for_p, scenario_seeds

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "exp06_stage4_noise_shift.py"
CFG = DatasetConfig()


def test_scenarios_are_well_formed():
    sc = build_scenarios()
    assert len({s["name"] for s in sc}) == len(sc) == 13
    assert {region_for_p(p) for p in TRAINED_P} == {"trained"}
    assert region_for_p(0.03) == "interpolation" and region_for_p(0.005) == "extrapolation"
    assert region_for_p(0.2) == "extrapolation"
    assert all(0 <= s["p"] <= 1 and 0 <= s["q"] <= 1 for s in sc)


def test_shift_seeds_are_independent_of_the_dataset_seeds():
    assert SHIFT_SEED != CFG.master_seed
    shifted = generate_split(DatasetConfig(p_values=(0.05,)), "train", 3000, scenario_seeds(1)[0])
    train_seed = np.random.SeedSequence(CFG.master_seed).spawn(4)[0]
    original = generate_split(DatasetConfig(p_values=(0.05,)), "train", 3000, train_seed)
    match = (shifted["meta_realization"] == original["meta_realization"]).all(axis=1).mean()
    assert match < 0.9                                          # identical streams would give 1.0


def test_scenario_generation_is_reproducible():
    sc = build_scenarios()[0]
    a = run_scenario(sc, 2000, scenario_seeds(13)[0], {"lookup": LookupDecoder()}, CFG)
    b = run_scenario(sc, 2000, scenario_seeds(13)[0], {"lookup": LookupDecoder()}, CFG)
    assert a == b


def test_run_scenario_agrees_with_exact_analysis():
    sc = {"name": "p=0.1", "axis": "p", "p": 0.1, "q": 0.05, "region": "trained"}
    r = run_scenario(sc, 40_000, np.random.SeedSequence(5), {"lookup": LookupDecoder()}, CFG)
    assert r["analytic"]["oracle_aware"] <= r["analytic"]["lookup"]
    exp, n = r["analytic"]["lookup"], r["n"]
    sigma = (exp * (1 - exp) / n) ** 0.5
    assert abs(r["metrics"]["lookup"]["failure_rate"] - exp) < 5 * sigma
    assert set(r["gain_vs_lookup"]) == {"oracle_aware"}


def test_load_frozen_models_checks_the_recorded_selection(tmp_path):
    data = generate_split(CFG, "train", 2000, np.random.SeedSequence(2))
    params = {"max_depth": 3, "min_samples_leaf": 20, "class_weight": None}
    lr_params = {"C": 1.0, "class_weight": None}
    SklearnDecoder.fit("tree", "raw", params, data["X"], data["y"]).save(tmp_path / "exp04_tree")
    SklearnDecoder.fit("logreg", "raw", lr_params, data["X"], data["y"]).save(tmp_path / "exp04_logreg")
    import json
    record = {"selected_1se": {"tree": {"kind": "tree", "feature_set": "raw", "params": params},
                               "logreg": {"kind": "logreg", "feature_set": "raw", "params": lr_params}}}
    (tmp_path / "sel.json").write_text(json.dumps(record))
    models, hashes = load_frozen_models(tmp_path, tmp_path / "sel.json")
    assert set(models) == {"tree", "logreg"} and all(len(h) == 64 for h in hashes.values())
    record["selected_1se"]["tree"]["params"] = {**params, "max_depth": 9}
    (tmp_path / "sel.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_frozen_models(tmp_path, tmp_path / "sel.json")


def test_plot_series_writes_a_file(tmp_path):
    series = {"a": [(0.01, 0.1, 0.09, 0.11), (0.1, 0.2, 0.19, 0.21)], "b": [(0.01, 0.05, 0.04, 0.06), (0.1, 0.1, 0.09, 0.11)]}
    plot_series(series, tmp_path / "s.png", "x", "y", "t", xlog=True, ylog=True, shade=(0.02, 0.1), zero_line=True)
    assert (tmp_path / "s.png").stat().st_size > 2000


def test_stage4_script_opens_no_dataset_and_trains_nothing():
    text = SCRIPT.read_text()
    for forbidden in ("test.npz", "val.npz", "train.npz", "load_split(", "fit(", "candidate_grid"):
        assert forbidden not in text, forbidden