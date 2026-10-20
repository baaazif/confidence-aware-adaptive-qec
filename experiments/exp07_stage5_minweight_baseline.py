"""Stage 5: is the ML gain still there against a HISTORY-AWARE classical decoder?

Development on the validation split; confirmation on a FRESH dataset (new seed).
The test split is not opened, so its single-use status from stage 3 is preserved."""
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
from caqec.evaluation.decoder_metrics import evaluate, evaluate_by_group, paired_difference
from caqec.evaluation.frozen import load_frozen_models
from caqec.plotting.decoder_plots import plot_failure_rates, plot_paired_differences
from caqec.simulation.dataset import DatasetConfig, generate_split, load_split

CONFIRM_SEED = 20261021          # new seed family: differs from dataset (…019) and shift (…020)
CONFIRM_N_PER_P = 50_000

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data" / "exp02"))
ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--exp04-results", default=str(ROOT / "results" / "exp04_stage2_selection_val.json"))
ap.add_argument("--results-dir", default=str(ROOT / "results"))
ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
args = ap.parse_args()

cfg = DatasetConfig()
models, model_hashes = load_frozen_models(args.models_dir, args.exp04_results)
mw = MinWeightDecoder(cfg.rounds)                      # a-priori nominal p = q = 0.05 (equal weights)
decoders = {"lookup": LookupDecoder(), "minweight": mw, "oracle": ExactDecoder(cfg),
            "logreg": models["logreg"], "tree": models["tree"]}
names = list(decoders)
groups = ["all"] + [str(p) for p in cfg.p_values]


def run(X, y, p_col):
    preds = {k: d.predict_flip(X) for k, d in decoders.items()}
    by_group = {}
    for k in names:
        per_p = {str(p): m for p, m in evaluate_by_group(y, preds[k], p_col).items()}
        by_group[k] = {"all": evaluate(y, preds[k]), **per_p}
    paired = {"all": {k: paired_difference(y, preds["lookup"], preds[k]) for k in names[1:]}}
    for p in cfg.p_values:
        m = np.isclose(p_col, p)
        paired[str(p)] = {k: paired_difference(y[m], preds["lookup"][m], preds[k][m]) for k in names[1:]}
    head = {k: paired_difference(y, preds["minweight"], preds[k]) for k in ("logreg", "tree", "oracle")}
    return preds, by_group, paired, head


def show(title, y, by_group, paired, head):
    print(f"{title}: {len(y)} rows, y=1 fraction {y.mean():.4f}")
    print(f"  {'decoder':10s} " + " ".join(f"{('all' if g == 'all' else 'p=' + g):24s}" for g in groups))
    for k in names:
        print(f"  {k:10s} " + " ".join(
            f"{by_group[k][g]['failure_rate']:.4f} [{by_group[k][g]['ci_low']:.4f},{by_group[k][g]['ci_high']:.4f}]".ljust(25)
            for g in groups))
    print("  gain vs lookup = failure(lookup) - failure(model), overall:")
    for k in names[1:]:
        d = paired["all"][k]
        print(f"    {k:10s} {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  McNemar p={d['mcnemar_p']:.3g}")
    print("  head-to-head vs minweight = failure(minweight) - failure(model), overall (positive = model better):")
    for k, d in head.items():
        print(f"    {k:10s} {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  McNemar p={d['mcnemar_p']:.3g}")


# ---- A. validation split (development) ----
val = load_split(Path(args.data_dir) / "val.npz")
_, v_by, v_paired, v_head = run(val["X"], val["y"], val["meta_p"])
show("A. VALIDATION split", val["y"], v_by, v_paired, v_head)

print("\n  sensitivity of the classical decoder to its NOMINAL p (validation, informational; q_nom = 0.05)")
for p_nom in (0.02, 0.05, 0.10):
    d = MinWeightDecoder(cfg.rounds, p_nom=p_nom, q_nom=0.05)
    pred = d.predict_flip(val["X"])
    per = evaluate_by_group(val["y"], pred, val["meta_p"])
    cells = " ".join(f"p={p}: {per[p]['failure_rate']:.4f}" for p in cfg.p_values)
    print(f"    p_nom={p_nom:<5} overall {evaluate(val['y'], pred)['failure_rate']:.4f} | {cells}")

# ---- B. fresh confirmation set ----
conf = generate_split(cfg, "train", CONFIRM_N_PER_P, np.random.SeedSequence(CONFIRM_SEED))  # name only picks an id offset
c_preds, c_by, c_paired, c_head = run(conf["X"], conf["y"], conf["meta_p"])
print()
show("B. FRESH confirmation set (new seed)", conf["y"], c_by, c_paired, c_head)

res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "exp07_stage5_minweight.json", "w") as f:
    json.dump({"confirm_seed": CONFIRM_SEED, "confirm_rows": int(len(conf["y"])), "model_sha256": model_hashes,
               "validation": {"by_group": v_by, "gain_vs_lookup": v_paired, "head_to_head_vs_minweight": v_head},
               "confirmation": {"by_group": c_by, "gain_vs_lookup": c_paired, "head_to_head_vs_minweight": c_head}},
              f, indent=2)
plot_failure_rates(c_by, groups, fig_dir / "exp07_confirm_failure_rates.png",
                   "Fresh confirmation set: failure rate with 95% intervals")
plot_paired_differences(c_paired, names[1:], groups, fig_dir / "exp07_confirm_gain_vs_lookup.png",
                        "Fresh confirmation set: improvement over the lookup decoder")
print(f"\nSaved results/exp07_stage5_minweight.json and 2 figures in {fig_dir}")