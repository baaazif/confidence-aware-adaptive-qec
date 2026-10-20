"""Stage 4: generalization of the FROZEN decoders across noise levels, on FRESH simulated data.
Opens no dataset files (neither train, val nor test); nothing is retrained or retuned."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import sklearn

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.evaluation.frozen import load_frozen_models
from caqec.evaluation.shift_eval import run_scenario
from caqec.plotting.shift_plots import plot_series
from caqec.simulation.dataset import DatasetConfig
from caqec.simulation.shift import (BASE_Q, Q_SWEEP_P, SHIFT_SEED, TRAINED_P, build_scenarios, scenario_seeds)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200_000, help="fresh rows per scenario")
ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--exp04-results", default=str(ROOT / "results" / "exp04_stage2_selection_val.json"))
ap.add_argument("--results-dir", default=str(ROOT / "results"))
ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
args = ap.parse_args()

base_cfg = DatasetConfig()
models, model_hashes = load_frozen_models(args.models_dir, args.exp04_results)
decoders = {"lookup": LookupDecoder(), "oracle_mixture": ExactDecoder(base_cfg),
            "logreg": models["logreg"], "tree": models["tree"]}
names = ["lookup", "oracle_mixture", "logreg", "tree", "oracle_aware"]
gain_names = ["oracle_mixture", "logreg", "tree", "oracle_aware"]

scenarios = build_scenarios()
results = [run_scenario(sc, args.n, seed, decoders, base_cfg)
           for sc, seed in zip(scenarios, scenario_seeds(len(scenarios)))]
print(f"{len(results)} scenarios x {args.n} fresh rows; frozen models trained at p in {TRAINED_P}, q={BASE_Q}\n")


def verdict(g):
    return "helps" if g["ci_low"] > 0 else "HURTS" if g["ci_high"] < 0 else "~"


def table(rows, label_key):
    print(f"{label_key:>6s} {'region':>13s} | analytic: lookup  aware | failure rate: " + " ".join(f"{n:>14s}" for n in names))
    for r in rows:
        sc, m = r["scenario"], r["metrics"]
        print(f"{sc[label_key]:>6} {sc['region']:>13s} | {r['analytic']['lookup']:>16.5f} {r['analytic']['oracle_aware']:>6.5f} | "
              + " ".join(f"{m[n]['failure_rate']:>14.5f}" for n in names))
    print(f"\n{label_key:>6s} gain = failure(lookup) - failure(model), 95% CI, verdict (CI excludes 0; exploratory)")
    for r in rows:
        cells = []
        for n in gain_names:
            g = r["gain_vs_lookup"][n]
            cells.append(f"{n}: {g['diff']:+.4f} [{g['ci_low']:+.4f},{g['ci_high']:+.4f}] {verdict(g)}")
        print(f"{r['scenario'][label_key]:>6}  " + " | ".join(cells))


p_rows = sorted([r for r in results if r["scenario"]["q"] == BASE_Q], key=lambda r: r["scenario"]["p"])
q_rows = sorted([r for r in results if r["scenario"]["p"] == Q_SWEEP_P], key=lambda r: r["scenario"]["q"])
print("=== p sweep (q = 0.05) ===")
table(p_rows, "p")
print(f"\n=== q sweep (p = {Q_SWEEP_P}) ===")
table(q_rows, "q")

res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "exp06_stage4_shift.json", "w") as f:
    json.dump({"shift_seed": SHIFT_SEED, "n_per_scenario": args.n, "trained_p": list(TRAINED_P), "trained_q": BASE_Q,
               "model_sha256": model_hashes, "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
               "scenarios": results}, f, indent=2)
with open(res_dir / "exp06_stage4_shift.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "axis", "p", "q", "region", "decoder", "failure_rate", "ci_low", "ci_high",
                "analytic_lookup", "analytic_oracle_aware"])
    for r in results:
        sc = r["scenario"]
        for n in names:
            m = r["metrics"][n]
            w.writerow([sc["name"], sc["axis"], sc["p"], sc["q"], sc["region"], n, m["failure_rate"],
                        m["ci_low"], m["ci_high"], r["analytic"]["lookup"], r["analytic"]["oracle_aware"]])

point = lambda r, m: (r["scenario"]["p"], m["failure_rate"], m["ci_low"], m["ci_high"])
plot_series({n: [point(r, r["metrics"][n]) for r in p_rows] for n in names}, fig_dir / "exp06_failure_vs_p.png",
            "physical flip probability p (q = 0.05)", "logical failure rate (fresh data)",
            "Frozen decoders under noise shift", xlog=True, ylog=True, shade=(min(TRAINED_P), max(TRAINED_P)))
gain = lambda r, n, xkey: (r["scenario"][xkey], r["gain_vs_lookup"][n]["diff"], r["gain_vs_lookup"][n]["ci_low"],
                           r["gain_vs_lookup"][n]["ci_high"])
plot_series({n: [gain(r, n, "p") for r in p_rows] for n in gain_names}, fig_dir / "exp06_gain_vs_p.png",
            "physical flip probability p (q = 0.05)", "failure(lookup) - failure(model)\n(positive = model better)",
            "Improvement over lookup under noise shift", xlog=True, shade=(min(TRAINED_P), max(TRAINED_P)), zero_line=True)
plot_series({n: [gain(r, n, "q") for r in q_rows] for n in gain_names}, fig_dir / "exp06_gain_vs_q.png",
            f"measurement-error probability q (p = {Q_SWEEP_P}; trained at q = {BASE_Q})",
            "failure(lookup) - failure(model)\n(positive = model better)",
            "Improvement over lookup under measurement-noise shift", zero_line=True)
print(f"\nSaved results/exp06_stage4_shift.json/.csv and 3 figures in {fig_dir}")