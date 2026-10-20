"""Stage 6: how trustworthy are the decoders' probabilities, and does rejecting low-confidence shots help?

Measurement only: nothing is fitted or recalibrated. Uses the validation split and a FRESH dataset
(new seed). The test split is not opened."""
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
from caqec.decoders.exact import ExactDecoder
from caqec.decoders.minweight import MinWeightDecoder
from caqec.evaluation.calibration import (at_coverage, expected_calibration_error, reliability_table,
                                          risk_coverage, truth_gap)
from caqec.evaluation.decoder_metrics import evaluate
from caqec.evaluation.frozen import load_frozen_models
from caqec.plotting.calibration_plots import plot_reliability, plot_risk_coverage
from caqec.simulation.dataset import DatasetConfig, generate_split, load_split

CALIB_SEED = 20261022            # new seed family
FRESH_N_PER_P = 50_000
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
prob_decoders = {"oracle": oracle, "tree": models["tree"], "logreg": models["logreg"]}
hard_decoders = {"lookup": LookupDecoder(), "minweight": MinWeightDecoder(cfg.rounds)}

val = load_split(Path(args.data_dir) / "val.npz")
fresh = generate_split(cfg, "train", FRESH_N_PER_P, np.random.SeedSequence(CALIB_SEED))  # name only picks an id offset
datasets = {"validation": val, "fresh": fresh}

out = {"calib_seed": CALIB_SEED, "model_sha256": model_hashes, "datasets": {}}
curves_for_plot, tables_for_plot = {}, {}
for ds_name, ds in datasets.items():
    X, y = ds["X"], ds["y"]
    p_true = oracle.predict_proba(X)             # exact posterior of y given the features (mixture noise)
    res = {"n": int(len(y)), "scores": {}, "reliability": {}, "risk_coverage": {}, "hard_reference": {}}
    print(f"\n=== {ds_name}: {len(y)} rows, y=1 fraction {y.mean():.4f} ===")
    print(f"{'decoder':8s} {'fail':>7s} {'brier':>8s} {'log_loss':>9s} {'ECE':>7s} {'gap to truth (mean|.|, rms)':>30s}")
    for name, d in prob_decoders.items():
        p = d.predict_proba(X)
        pred = d.predict_flip(X)
        m = evaluate(y, pred, p)
        gap = truth_gap(p, p_true)
        ece = expected_calibration_error(y, p)
        res["scores"][name] = {"failure_rate": m["failure_rate"], "brier": m["brier"], "log_loss": m["log_loss"],
                               "ece": ece, "truth_gap": gap}
        res["reliability"][name] = reliability_table(y, p)
        pts = risk_coverage(y, pred, d.confidence(X))
        res["risk_coverage"][name] = pts
        print(f"{name:8s} {m['failure_rate']:7.4f} {m['brier']:8.4f} {m['log_loss']:9.4f} {ece:7.4f} "
              f"{gap['mean_abs']:>16.4f} {gap['rms']:>13.4f}")
    for name, d in hard_decoders.items():
        res["hard_reference"][name] = evaluate(y, d.predict_flip(X))["failure_rate"]

    print("\nReliability (observed frequency of y=1 vs mean prediction, per bin):")
    for name in prob_decoders:
        cells = [f"[{r['low']:.1f},{r['high']:.1f}) n={r['n']:>6d} pred={r['mean_pred']:.3f} obs={r['observed']:.3f}"
                 for r in res["reliability"][name] if r["n"] > 0]
        print(f"  {name}:")
        for c in cells:
            print("    " + c)

    print("\nRisk-coverage: failure rate among ACCEPTED shots when accepting the most confident >= coverage target")
    print(f"  (no-confidence references at full coverage: " +
          ", ".join(f"{k} {v:.4f}" for k, v in res["hard_reference"].items()) + "; random rejection leaves the rate unchanged)")
    print(f"  {'target':>7s} " + " ".join(f"{n:>34s}" for n in prob_decoders))
    for t in TARGETS:
        cells = []
        for name in prob_decoders:
            q = at_coverage(res["risk_coverage"][name], t)
            cells.append(f"cov {q['coverage']:.3f} fail {q['failure_rate']:.4f} [{q['ci_low']:.4f},{q['ci_high']:.4f}]")
        print(f"  {t:>7.2f} " + " ".join(f"{c:>34s}" for c in cells))
    out["datasets"][ds_name] = res
    curves_for_plot[ds_name] = res["risk_coverage"]
    tables_for_plot[ds_name] = res["reliability"]

res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "exp08_stage6_calibration_risk.json", "w") as f:
    json.dump(out, f, indent=2)
plot_reliability(tables_for_plot["fresh"], fig_dir / "exp08_reliability_fresh.png", "Reliability on the fresh set")
plot_risk_coverage(curves_for_plot["fresh"], fig_dir / "exp08_risk_coverage_fresh.png",
                   "Rejecting low-confidence shots (fresh set)", reference=out["datasets"]["fresh"]["hard_reference"])
print(f"\nSaved results/exp08_stage6_calibration_risk.json and 2 figures in {fig_dir}")