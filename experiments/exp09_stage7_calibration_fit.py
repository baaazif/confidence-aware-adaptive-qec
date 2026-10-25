"""Stage 7: calibrate the tree and logistic-regression probabilities and check on NEW fresh data.

Calibrators are fitted on the VALIDATION split only. Evaluation uses a fresh dataset (new seed).
The base models are frozen and the test split is not opened."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.calibrated import CalibratedDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.decoders.minweight import MinWeightDecoder
from caqec.evaluation.calibration import (at_coverage, expected_calibration_error, reliability_table,
                                          risk_coverage, truth_gap)
from caqec.evaluation.decoder_metrics import evaluate, paired_difference
from caqec.evaluation.frozen import load_frozen_models
from caqec.evaluation.stats import wilson_interval
from caqec.plotting.calibration_plots import plot_reliability, plot_risk_coverage
from caqec.simulation.dataset import DatasetConfig, generate_split, load_split

EVAL_SEED = 20261023            # new seed family
EVAL_N_PER_P = 50_000
TARGETS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data" / "exp02"))
ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--exp04-results", default=str(ROOT / "results" / "exp04_stage2_selection_val.json"))
ap.add_argument("--results-dir", default=str(ROOT / "results"))
ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
args = ap.parse_args()

cfg = DatasetConfig()
models, model_hashes = load_frozen_models(args.models_dir, args.exp04_results)
oracle = ExactDecoder(cfg)
val = load_split(Path(args.data_dir) / "val.npz")                      # calibrators see ONLY this
fresh = generate_split(cfg, "train", EVAL_N_PER_P, np.random.SeedSequence(EVAL_SEED))  # name only picks an id offset
X, y = fresh["X"], fresh["y"]

decoders = {"oracle": oracle}
for base_name in ("tree", "logreg"):
    base = models[base_name]
    decoders[f"{base_name}_raw"] = base
    for method in ("sigmoid", "isotonic"):
        decoders[f"{base_name}_{method}"] = CalibratedDecoder.fit(base, val["X"], val["y"], method)
groups = {"tree": ["tree_raw", "tree_sigmoid", "tree_isotonic", "oracle"],
          "logreg": ["logreg_raw", "logreg_sigmoid", "logreg_isotonic", "oracle"]}
hard = {"lookup": LookupDecoder(), "minweight": MinWeightDecoder(cfg.rounds)}

p_true = oracle.predict_proba(X)
out = {"eval_seed": EVAL_SEED, "n_eval": int(len(y)), "calibration_rows": int(len(val["y"])),
       "model_sha256": model_hashes, "scores": {}, "reliability": {}, "risk_coverage": {}, "vs_raw": {}}
preds, probas = {}, {}
print(f"Calibrators fitted on {len(val['y'])} validation rows; evaluated on {len(y)} NEW fresh rows (y=1: {y.mean():.4f})\n")
print(f"{'decoder':16s} {'fail':>7s} {'brier':>8s} {'log_loss':>9s} {'ECE':>7s} {'gap mean|.|':>12s} {'gap rms':>8s} {'levels':>7s}")
for name, d in decoders.items():
    probas[name], preds[name] = d.predict_proba(X), d.predict_flip(X)
    m = evaluate(y, preds[name], probas[name])
    gap, ece = truth_gap(probas[name], p_true), expected_calibration_error(y, probas[name])
    levels = int(len(np.unique(d.confidence(X))))
    out["scores"][name] = {"failure_rate": m["failure_rate"], "brier": m["brier"], "log_loss": m["log_loss"],
                           "ece": ece, "truth_gap": gap, "distinct_confidence_levels": levels}
    out["reliability"][name] = reliability_table(y, probas[name])
    out["risk_coverage"][name] = risk_coverage(y, preds[name], d.confidence(X))
    print(f"{name:16s} {m['failure_rate']:7.4f} {m['brier']:8.4f} {m['log_loss']:9.4f} {ece:7.4f} "
          f"{gap['mean_abs']:12.4f} {gap['rms']:8.4f} {levels:7d}")

print("\nEffect of calibration on the hard decision: failure(raw) - failure(calibrated); positive = calibration helped")
for base_name in ("tree", "logreg"):
    for method in ("sigmoid", "isotonic"):
        d = paired_difference(y, preds[f"{base_name}_raw"], preds[f"{base_name}_{method}"])
        out["vs_raw"][f"{base_name}_{method}"] = d
        print(f"  {base_name}+{method:8s} {d['diff']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]  McNemar p={d['mcnemar_p']:.3g}")

# control: reject RANDOM shots instead of low-confidence ones (tree_raw decisions)
rng = np.random.default_rng(0)
wrong = preds["tree_raw"] != y
random_control = {}
for t in TARGETS:
    keep = rng.random(len(y)) < t
    k, a = int(wrong[keep].sum()), int(keep.sum())
    lo, hi = wilson_interval(k, a)
    random_control[t] = {"coverage": a / len(y), "failure_rate": k / a, "ci_low": lo, "ci_high": hi}
out["random_control_tree_raw"] = {str(t): v for t, v in random_control.items()}

for base_name, names in groups.items():
    print(f"\nRisk-coverage ({base_name} family): failure among ACCEPTED shots; attempts per accepted shot = 1/coverage")
    print(f"  no-confidence references at full coverage: " +
          ", ".join(f"{k} {evaluate(y, d.predict_flip(X))['failure_rate']:.4f}" for k, d in hard.items()))
    print(f"  {'target':>6s} " + " ".join(f"{n:>30s}" for n in names) + f" {'random reject (tree_raw)':>28s}")
    for t in TARGETS:
        cells = []
        for n in names:
            q = at_coverage(out["risk_coverage"][n], t)
            cells.append(f"cov {q['coverage']:.3f} fail {q['failure_rate']:.4f}")
        r = random_control[t]
        print(f"  {t:>6.2f} " + " ".join(f"{c:>30s}" for c in cells) + f" {'cov %.3f fail %.4f' % (r['coverage'], r['failure_rate']):>28s}")

res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "exp09_stage7_calibration.json", "w") as f:
    json.dump(out, f, indent=2)
for base_name, names in groups.items():
    plot_reliability({n: out["reliability"][n] for n in names}, fig_dir / f"exp09_reliability_{base_name}.png",
                     f"Reliability on new fresh data: {base_name} before and after calibration")
plot_risk_coverage({n: out["risk_coverage"][n] for n in decoders}, fig_dir / "exp09_risk_coverage.png",
                   "Rejecting low-confidence shots: raw vs calibrated confidence",
                   reference={k: evaluate(y, d.predict_flip(X))["failure_rate"] for k, d in hard.items()})
print(f"\nSaved results/exp09_stage7_calibration.json and 3 figures in {fig_dir}")