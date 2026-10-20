"""Stage 2: train sklearn decoders on TRAIN, select on VALIDATION. The test split is never opened."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.tree import export_text

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.decoders.selection import candidate_grid, fit_and_score, select_best, select_one_se
from caqec.decoders.sklearn_decoder import SklearnDecoder, feature_names_for
from caqec.evaluation.decoder_metrics import evaluate, evaluate_by_group, paired_difference
from caqec.simulation.dataset import DatasetConfig, load_split

SEED = 0
cfg = DatasetConfig()
train = load_split(ROOT / "data" / "exp02" / "train.npz")
val = load_split(ROOT / "data" / "exp02" / "val.npz")
Xtr, ytr = train["X"], train["y"]            # only X and y are ever given to a model
Xva, yva, p_val = val["X"], val["y"], val["meta_p"]  # meta_p only groups the printed results
print(f"train {len(ytr)} rows (y=1: {ytr.mean():.4f}), val {len(yva)} rows (y=1: {yva.mean():.4f})")

grid = candidate_grid()
results = fit_and_score(grid, Xtr, ytr, Xva, yva, seed=SEED)
print(f"fitted {len(results)} candidates on train, scored on validation\n")

print("Class imbalance: best validation result per family and class_weight")
print(f"  {'family':8s} {'class_weight':12s} {'failure [95% CI]':24s} {'recall_y1':>9s} {'FPR':>7s}  params")
for kind in ("logreg", "tree"):
    for cw in (None, "balanced"):
        pool = [r for r in results if r["candidate"]["kind"] == kind and r["candidate"]["params"]["class_weight"] == cw]
        r = min(pool, key=lambda r: (r["val"]["failure_rate"], r["index"]))
        v = r["val"]
        print(f"  {kind:8s} {str(cw):12s} {v['failure_rate']:.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]   "
              f"{v['recall_y1']:>9.3f} {v['false_positive_rate']:>7.4f}  {r['candidate']['feature_set']} {r['candidate']['params']}")

selected = {}
print("\nSelection (validation only): lowest failure vs 1-SE rule (simplest within one standard error)")
for kind in ("logreg", "tree"):
    b, s = select_best(results, kind), select_one_se(results, kind)
    selected[kind] = s
    print(f"  {kind}: best  {b['val']['failure_rate']:.4f}  {b['candidate']['feature_set']} {b['candidate']['params']}")
    print(f"  {kind}: 1-SE  {s['val']['failure_rate']:.4f}  {s['candidate']['feature_set']} {s['candidate']['params']}  <- used")

lookup, exact = LookupDecoder(), ExactDecoder(cfg)
decoders = {"lookup": lookup, "exact(oracle)": exact,
            "logreg": selected["logreg"]["decoder"], "tree": selected["tree"]["decoder"]}
preds = {k: d.predict_flip(Xva) for k, d in decoders.items()}
probas = {k: d.predict_proba(Xva) for k, d in decoders.items()}
by_p = {k: evaluate_by_group(yva, preds[k], p_val, probas[k]) for k in decoders}

print(f"\nValidation failure rate [95% CI]  (analytic: lookup {exact.expected_failure_rate(np.zeros(len(exact.joint), dtype=int)):.5f}, "
      f"optimum {exact.expected_failure_rate():.5f})")
print(f"  {'decoder':14s} {'all':24s} " + " ".join(f"{'p=' + str(p):24s}" for p in cfg.p_values))
for k in decoders:
    o = evaluate(yva, preds[k], probas[k])
    cells = [f"{o['failure_rate']:.4f} [{o['ci_low']:.4f},{o['ci_high']:.4f}]"] + \
            [f"{by_p[k][p]['failure_rate']:.4f} [{by_p[k][p]['ci_low']:.4f},{by_p[k][p]['ci_high']:.4f}]" for p in cfg.p_values]
    print(f"  {k:14s} " + " ".join(f"{c:24s}" for c in cells))

print("\nPaired difference vs lookup, failure(lookup) - failure(model) (positive = model better)")
paired = {}
for k in ("exact(oracle)", "logreg", "tree"):
    d = paired_difference(yva, preds["lookup"], preds[k])
    paired[k] = d
    print(f"  {k:14s} {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  McNemar p={d['mcnemar_p']:.3g}")
gap = evaluate(yva, preds["lookup"])["failure_rate"] - evaluate(yva, preds["exact(oracle)"])["failure_rate"]
for k in ("logreg", "tree"):
    got = evaluate(yva, preds["lookup"])["failure_rate"] - evaluate(yva, preds[k])["failure_rate"]
    print(f"  share of validation headroom (lookup -> oracle) captured by {k}: {got / gap:.2f}  (informational, noisy)")

lr = decoders["logreg"]
names = feature_names_for(lr.feature_set, lr.rounds)
coef = lr.estimator.coef_[0]
print(f"\nSelected logistic regression ({lr.feature_set}): intercept {lr.estimator.intercept_[0]:+.3f}, 8 largest |coefficients|")
for i in np.argsort(-np.abs(coef))[:8]:
    print(f"  {names[i]:28s} {coef[i]:+.3f}")
print("\nSelected decision tree (top 2 levels):")
print(export_text(decoders["tree"].estimator, feature_names=feature_names_for("raw", decoders["tree"].rounds), max_depth=2))

(ROOT / "models").mkdir(exist_ok=True)
for kind in ("logreg", "tree"):
    path = ROOT / "models" / f"exp04_{kind}"
    decoders[kind].save(path)
    again = SklearnDecoder.load(path)
    ok = np.array_equal(again.predict_proba(Xva), probas[kind])
    print(f"saved models/exp04_{kind}.joblib/.json; reload gives identical probabilities: {ok}")
    assert ok

out = {"seed": SEED, "n_train": int(len(ytr)), "n_val": int(len(yva)),
       "grid": [{"index": r["index"], **r["candidate"], "val": r["val"]} for r in results],
       "selected_1se": {k: {"index": v["index"], **v["candidate"]} for k, v in selected.items()},
       "val_overall": {k: evaluate(yva, preds[k], probas[k]) for k in decoders},
       "val_by_p": {k: {str(p): m for p, m in by_p[k].items()} for k in decoders},
       "paired_lookup_minus_model": paired}
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "exp04_stage2_selection_val.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved results/exp04_stage2_selection_val.json")