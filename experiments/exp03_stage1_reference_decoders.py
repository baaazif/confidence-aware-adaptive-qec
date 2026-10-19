"""Stage 1: reference decoders evaluated on the VALIDATION split only.
The test split is deliberately never opened here."""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.evaluation.decoder_metrics import evaluate, evaluate_by_group, paired_difference
from caqec.simulation.dataset import DatasetConfig, load_split

cfg = DatasetConfig()
val = load_split(ROOT / "data" / "exp02" / "val.npz")
X, y, p_col = val["X"], val["y"], val["meta_p"]  # meta_p is used ONLY to group the results

lookup, exact = LookupDecoder(), ExactDecoder(cfg)
preds = {d.name: d.predict_flip(X) for d in (lookup, exact)}
probas = {"lookup": lookup.predict_proba(X), "exact": exact.predict_proba(X)}

zeros = np.zeros(len(exact.joint), dtype=int)
analytic = {"lookup": exact.expected_failure_rate(zeros), "exact": exact.expected_failure_rate()}
mixture_rule = (exact._proba > 0.5).astype(int)  # the p-agnostic decision rule
analytic_p = {}
for p in cfg.p_values:
    ex_p = ExactDecoder(replace(cfg, p_values=(p,)))  # noise model with this single p
    analytic_p[p] = {"lookup": ex_p.expected_failure_rate(zeros),
                     "exact_mixture_rule": ex_p.expected_failure_rate(mixture_rule),
                     "exact_p_aware": ex_p.expected_failure_rate()}

results = {"split": "val", "n": int(len(y)), "overall": {}, "by_p": {}, "analytic_overall": analytic,
           "analytic_by_p": {str(k): v for k, v in analytic_p.items()}, "paired_lookup_minus_exact": {}}
print(f"Validation split only: {len(y)} rows, y=1 fraction {y.mean():.4f}\n")
print("Exact (analytic) failure probabilities for the p-mixture:",
      {k: round(v, 5) for k, v in analytic.items()})

for name in ("lookup", "exact"):
    results["overall"][name] = evaluate(y, preds[name], probas[name])
    results["by_p"][name] = {str(k): v for k, v in evaluate_by_group(y, preds[name], p_col, probas[name]).items()}
results["paired_lookup_minus_exact"]["overall"] = paired_difference(y, preds["lookup"], preds["exact"])
for p in cfg.p_values:
    m = np.isclose(p_col, p)
    results["paired_lookup_minus_exact"][str(p)] = paired_difference(y[m], preds["lookup"][m], preds["exact"][m])

print(f"\n{'decoder':8s} {'p':>6s} {'failure rate [95% CI]':28s} {'analytic':>9s} {'recall_y1':>9s} {'FPR':>7s} {'pred_y1':>8s}")
for name in ("lookup", "exact"):
    key = "lookup" if name == "lookup" else "exact_mixture_rule"
    rows = [("all", results["overall"][name], analytic[name])]
    rows += [(str(p), results["by_p"][name][str(p)], analytic_p[p][key]) for p in cfg.p_values]
    for label, r, a in rows:
        rec = "n/a" if r["recall_y1"] is None else f"{r['recall_y1']:.3f}"
        fpr = "n/a" if r["false_positive_rate"] is None else f"{r['false_positive_rate']:.4f}"
        print(f"{name:8s} {label:>6s} {r['failure_rate']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]   {a:>9.5f} {rec:>9s} {fpr:>7s} {r['predicted_y1']:>8d}")

print("\nAnalytic failure probability per p (exact, no sampling error):")
print(f"  {'p':>5s} {'lookup':>9s} {'exact, mixture rule':>20s} {'exact, knows p':>15s}")
for p in cfg.p_values:
    a = analytic_p[p]
    print(f"  {p:>5} {a['lookup']:>9.5f} {a['exact_mixture_rule']:>20.5f} {a['exact_p_aware']:>15.5f}")

print("\nPaired difference, failure(lookup) - failure(exact), same rows:")
for key, d in results["paired_lookup_minus_exact"].items():
    print(f"  {key:>7s}: {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
          f"lookup-only-wrong={d['a_wrong_b_right']} exact-only-wrong={d['a_right_b_wrong']} McNemar p={d['mcnemar_p']:.3g}")

out = ROOT / "results" / "exp03_stage1_reference_val.json"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out.relative_to(ROOT)}")