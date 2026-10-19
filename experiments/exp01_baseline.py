import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # write PNG files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit_aer

from caqec.evaluation.stats import wilson_interval
from caqec.evaluation.theory import any_physical_flip, protected_failure, unprotected_failure
from caqec.simulation.baseline import run_protected, run_unprotected

P_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SHOTS = 200_000
BASE_SEED = 1234

results_dir, figures_dir = ROOT / "results", ROOT / "figures"
results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

print(f"qiskit {qiskit.__version__}, qiskit-aer {qiskit_aer.__version__}, "
      f"shots per point = {SHOTS}, base seed = {BASE_SEED}\n")

rows = []
for i, p in enumerate(P_VALUES):
    unp = run_unprotected(p, SHOTS, seed=BASE_SEED + 2 * i)      # independent seed per run
    pro = run_protected(p, SHOTS, seed=BASE_SEED + 2 * i + 1)
    for arm, res, theory in (("unprotected", unp, unprotected_failure(p)),
                             ("protected", pro, protected_failure(p))):
        lo, hi = wilson_interval(res["failures"], SHOTS)
        rows.append({"p": p, "arm": arm, "shots": SHOTS, "failures": res["failures"],
                     "rate": res["failures"] / SHOTS, "ci_low": lo, "ci_high": hi,
                     "theory": theory})
    flip_lo, flip_hi = wilson_interval(pro["any_flip"], SHOTS)
    rows.append({"p": p, "arm": "protected_any_physical_flip", "shots": SHOTS,
                 "failures": pro["any_flip"], "rate": pro["any_flip"] / SHOTS,
                 "ci_low": flip_lo, "ci_high": flip_hi, "theory": any_physical_flip(p)})

csv_path = results_dir / "exp01_baseline.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# ---- printed table ----
def get(p, arm):
    return next(r for r in rows if r["p"] == p and r["arm"] == arm)

def fmt(r):
    inside = "in CI" if r["ci_low"] <= r["theory"] <= r["ci_high"] else "OUTSIDE CI"
    return (f"{r['rate']:.5f} [{r['ci_low']:.5f},{r['ci_high']:.5f}] "
            f"theory {r['theory']:.5f} {inside}")

for p in P_VALUES:
    print(f"p = {p}")
    print("  unprotected  :", fmt(get(p, "unprotected")))
    print("  protected    :", fmt(get(p, "protected")))
    print("  >=1 phys flip:", fmt(get(p, "protected_any_physical_flip")))
    pro_r, unp_r = get(p, "protected"), get(p, "unprotected")
    if pro_r["ci_high"] < unp_r["ci_low"]:
        verdict = "yes (protected is clearly lower)"
    elif pro_r["ci_low"] > unp_r["ci_high"]:
        verdict = "NO (protected is clearly higher)"
    else:
        verdict = "indistinguishable (intervals overlap)"
    print("  code helps?  :", verdict)

# ---- plot ----
fig, ax = plt.subplots(figsize=(7, 5.5))
grid = np.logspace(np.log10(0.004), np.log10(0.75), 200)
ax.plot(grid, [unprotected_failure(x) for x in grid], "k--", lw=1, label="theory: unprotected (p)")
ax.plot(grid, [protected_failure(x) for x in grid], "C0-", lw=1, label="theory: 3p^2 - 2p^3")
for arm, colour, marker, label in (("unprotected", "k", "o", "simulated: unprotected"),
                                   ("protected", "C0", "s", "simulated: 3-qubit code")):
    pts = [get(p, arm) for p in P_VALUES]
    y = np.array([r["rate"] for r in pts])
    yerr = np.array([[r["rate"] - r["ci_low"] for r in pts],
                     [r["ci_high"] - r["rate"] for r in pts]])
    ax.errorbar(P_VALUES, y, yerr=yerr, fmt=marker, color=colour, ms=5, capsize=3, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("physical bit-flip probability p")
ax.set_ylabel("logical failure rate")
ax.set_title(f"Experiment 01: {SHOTS:,} shots per point, 95% Wilson intervals")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig_path = figures_dir / "exp01_logical_vs_physical.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {csv_path.relative_to(ROOT)} and {fig_path.relative_to(ROOT)}")