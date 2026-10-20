"""Stage 3: the single evaluation of the FROZEN decoders on the TEST split.

Nothing is trained or selected here: models are loaded from disk and checked against the
selection recorded in stage 2. The script refuses to run twice (see --allow-rerun)."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import sklearn

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.decoders.sklearn_decoder import SklearnDecoder
from caqec.evaluation.decoder_metrics import evaluate, evaluate_by_group, paired_difference
from caqec.evaluation.protocol import file_sha256, git_state, single_use_guard, verify_split_checksum
from caqec.plotting.decoder_plots import plot_failure_rates, plot_paired_differences
from caqec.simulation.dataset import DatasetConfig, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data" / "exp02"))
ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--exp04-results", default=str(ROOT / "results" / "exp04_stage2_selection_val.json"))
ap.add_argument("--results-path", default=str(ROOT / "results" / "exp05_stage3_test.json"))
ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
ap.add_argument("--allow-rerun", action="store_true")
args = ap.parse_args()

results_path = Path(args.results_path)
single_use_guard(results_path, args.allow_rerun)          # 1. refuse a second look at the test split

cfg = DatasetConfig()
data_dir = Path(args.data_dir)
test = load_split(data_dir / "test.npz")
test_sha = verify_split_checksum(test, data_dir / "dataset_card.json", "test")   # 2. split is the one generated
X, y, p_col = test["X"], test["y"], test["meta_p"]         # meta_p only groups the results

selected = json.loads(Path(args.exp04_results).read_text())["selected_1se"]
models, model_files = {}, {}
for kind in ("logreg", "tree"):                            # 3. frozen models, checked against stage-2 selection
    base = Path(args.models_dir) / f"exp04_{kind}"
    m = SklearnDecoder.load(base)
    s = selected[kind]
    if (m.kind, m.feature_set, m.params) != (s["kind"], s["feature_set"], s["params"]):
        raise ValueError(f"saved {kind} model does not match the selection recorded in stage 2")
    models[kind] = m
    model_files[kind] = file_sha256(base.with_suffix(".joblib"))

exact = ExactDecoder(cfg)
decoders = {"lookup": LookupDecoder(), "exact(oracle)": exact, "logreg": models["logreg"], "tree": models["tree"]}
preds = {k: d.predict_flip(X) for k, d in decoders.items()}
probas = {k: d.predict_proba(X) for k, d in decoders.items()}
groups = ["all"] + [str(p) for p in cfg.p_values]

# Brier / log loss only for decoders whose probabilities mean something (not the hard lookup decoder)
proba_or_none = {k: probas[k] if d.provides_confidence else None for k, d in decoders.items()}
overall = {k: evaluate(y, preds[k], proba_or_none[k]) for k in decoders}
by_p = {k: {str(p): m for p, m in evaluate_by_group(y, preds[k], p_col, proba_or_none[k]).items()} for k in decoders}
by_group = {k: {"all": overall[k], **by_p[k]} for k in decoders}

compare = ("exact(oracle)", "logreg", "tree")
paired = {"all": {k: paired_difference(y, preds["lookup"], preds[k]) for k in compare}}
for p in cfg.p_values:
    m = np.isclose(p_col, p)
    paired[str(p)] = {k: paired_difference(y[m], preds["lookup"][m], preds[k][m]) for k in compare}
tree_vs_logreg = paired_difference(y, preds["logreg"], preds["tree"])
zeros = np.zeros(len(exact.joint), dtype=int)
analytic = {"lookup": exact.expected_failure_rate(zeros), "optimum": exact.expected_failure_rate()}

print(f"TEST split: {len(y)} rows, y=1 fraction {y.mean():.4f}, sha256 {test_sha[:12]}...")
print(f"analytic: lookup {analytic['lookup']:.5f}, optimum {analytic['optimum']:.5f}\n")
print(f"{'decoder':14s} " + " ".join(f"{('all' if g == 'all' else 'p=' + g):24s}" for g in groups))
for k in decoders:
    print(f"{k:14s} " + " ".join(
        f"{by_group[k][g]['failure_rate']:.4f} [{by_group[k][g]['ci_low']:.4f},{by_group[k][g]['ci_high']:.4f}]".ljust(25)
        for g in groups))
print(f"\n{'decoder':14s} {'accuracy':>9s} {'recall_y1':>9s} {'precision_y1':>12s} {'FPR':>7s} {'bal_acc':>8s} {'brier':>7s} {'log_loss':>8s}")
fmt = lambda v, w, d=3: f"{'n/a':>{w}s}" if v is None else f"{v:{w}.{d}f}"
for k in decoders:
    o = overall[k]
    print(f"{k:14s} {fmt(o['accuracy'], 9, 4)} {fmt(o['recall_y1'], 9)} {fmt(o['precision_y1'], 12)} "
          f"{fmt(o['false_positive_rate'], 7, 4)} {fmt(o['balanced_accuracy'], 8)} {fmt(o.get('brier'), 7, 4)} {fmt(o.get('log_loss'), 8, 4)}")

print("\nPaired difference, failure(lookup) - failure(model); positive = model better")
print("Headline contrasts (overall, 3 tests, Bonferroni-adjusted p); per-p rows are exploratory")
for g in groups:
    for k in compare:
        d = paired[g][k]
        adj = f"  adj p={min(1.0, 3 * d['mcnemar_p']):.3g}" if g == "all" else ""
        print(f"  {('all' if g == 'all' else 'p=' + g):8s} {k:14s} {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
              f"McNemar p={d['mcnemar_p']:.3g}{adj}")
print(f"  exploratory: failure(logreg) - failure(tree) overall: {tree_vs_logreg['diff']:+.4f} "
      f"[{tree_vs_logreg['ci_low']:+.4f}, {tree_vs_logreg['ci_high']:+.4f}]  McNemar p={tree_vs_logreg['mcnemar_p']:.3g}")

figures_dir = Path(args.figures_dir)
plot_failure_rates({k: by_group[k] for k in decoders}, groups, figures_dir / "exp05_test_failure_rates.png",
                   "Test split: logical failure rate with 95% intervals")
plot_paired_differences(paired, list(compare), groups, figures_dir / "exp05_test_paired_difference.png",
                        "Test split: improvement over the lookup decoder")

record = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "n_test": int(len(y)), "test_sha256": test_sha, "model_sha256": model_files,
          "selection_from_stage2": selected, "analytic": analytic,
          "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__, "python": sys.version.split()[0]},
          "git": git_state(ROOT), "overall": overall, "by_p": by_p, "paired_lookup_minus_model": paired,
          "logreg_minus_tree": tree_vs_logreg}
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(record, f, indent=2)
print(f"\nSaved {results_path} and two figures in {figures_dir}")