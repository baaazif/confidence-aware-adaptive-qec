"""Stage 9: confidence-aware adaptive controller vs fixed decoders, static ML and an adaptive baseline.

Headline settings are fixed IN ADVANCE (ControllerConfig defaults, CostModel defaults). Ablations and the
update-cost sensitivity are descriptive and are not used to choose the headline. Controllers see only
per-shot observables derived from syndromes. Labels are used only to count failures afterwards.
All adaptation costs are ASSUMED; decoder inference runtimes are MEASURED. Opens no dataset files."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import stats as sps

from caqec.control.controller import BASELINE_CONFIG, AdaptiveController, ControllerConfig, FixedController
from caqec.control.costs import CostModel
from caqec.control.harness import observables, run_controller
from caqec.control.knowledge import NoiseKnowledge
from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.minweight import MinWeightDecoder
from caqec.evaluation.frozen import load_frozen_models
from caqec.evaluation.stats import wilson_interval
from caqec.plotting.controller_plots import plot_ablation, plot_method_costs
from caqec.plotting.nonstationary_plots import plot_failure_timeseries
from caqec.simulation.dataset import DatasetConfig
from caqec.simulation.streams_q import generate_streams_q

EVAL_SEED, SENS_SEED = 20261040, 20261041
T, BIN, CHANGE = 6000, 250, 3000
cfg = DatasetConfig()
R, Q0 = cfg.rounds, cfg.q
t_ = np.arange(T)
SCENARIOS = {   # name -> (p_t, q_t, change_start, change_end)
    "stationary p=0.02": (np.full(T, .02), np.full(T, Q0), None, None),
    "stationary p=0.10": (np.full(T, .10), np.full(T, Q0), None, None),
    "abrupt 0.02->0.05": (np.where(t_ < CHANGE, .02, .05), np.full(T, Q0), CHANGE, CHANGE),
    "abrupt 0.02->0.10": (np.where(t_ < CHANGE, .02, .10), np.full(T, Q0), CHANGE, CHANGE),
    "abrupt 0.10->0.02": (np.where(t_ < CHANGE, .10, .02), np.full(T, Q0), CHANGE, CHANGE),
    "gradual 0.02->0.10": (.02 + .08 * np.clip((t_ - 2500) / 2000, 0, 1), np.full(T, Q0), 2500, 4500),
    "q-shift 0.05->0.15 (p=0.05)": (np.full(T, .05), np.where(t_ < CHANGE, Q0, .15), CHANGE, CHANGE),
}
P = ControllerConfig()
ADAPTIVE = {"adaptive_baseline": BASELINE_CONFIG, "proposed": P,
            "abl_no_guard": replace(P, name="abl_no_guard", use_guard=False),
            "abl_no_dwell": replace(P, name="abl_no_dwell", dwell=0),
            "abl_cost_blind": replace(P, name="abl_cost_blind", cost_aware=False),
            "abl_no_update": replace(P, name="abl_no_update", allow_update=False),
            "abl_nominal_q": replace(P, name="abl_nominal_q", estimate_q=False)}
FIXED = {"fixed_lookup": "lookup", "fixed_minweight": "minweight", "static_tree": "tree"}
MAIN = ["fixed_lookup", "fixed_minweight", "static_tree", "adaptive_baseline", "proposed"]
ABLATIONS = [m for m in ADAPTIVE if m.startswith("abl_")] + ["adaptive_baseline"]

ap = argparse.ArgumentParser()
ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--exp04-results", default=str(ROOT / "results" / "exp04_stage2_selection_val.json"))
ap.add_argument("--results-dir", default=str(ROOT / "results"))
ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
ap.add_argument("--trials", type=int, default=60)
args = ap.parse_args()
TR = args.trials

models, model_hashes = load_frozen_models(args.models_dir, args.exp04_results)
decoders = {"lookup": LookupDecoder(), "minweight": MinWeightDecoder(R), "tree": models["tree"], "logreg": models["logreg"]}
ALL_X = ((np.arange(2 ** (2 * R + 2))[:, None] >> np.arange(2 * R + 2)) & 1).astype(np.uint8)
K = NoiseKnowledge(R, {k: decoders[k].predict_flip(ALL_X) for k in ("lookup", "minweight", "tree")},
                   models["tree"].predict_proba(ALL_X))
costs = CostModel()


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float(x.mean()), float(x.mean())
    h = sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / len(x) ** 0.5
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, lo, hi = mean_ci(d)
    try:
        pv = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        pv = 1.0
    return {"diff": m, "ci_low": lo, "ci_high": hi, "wilcoxon_p": pv}


# ---------- A. validation of the confidence signal (exact, no sampling) ----------
print("A1. Is the tree's self-estimated failure (label-free) calibrated under noise shift?  (exact)")
print(f"   {'p':>6s} {'q':>5s} {'actual failure':>15s} {'self-estimated':>15s} {'ratio est/actual':>17s}")
conf_table = []
for q in (0.05, 0.15):
    for p in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        a, s = K.failure("tree", p, q), K.mean_selfp(p, q)
        conf_table.append({"p": p, "q": q, "actual": a, "self_estimated": s})
        print(f"   {p:>6} {q:>5} {a:>15.5f} {s:>15.5f} {s / a:>17.2f}")
print("\nA2. Noise estimators on simulated 300-shot windows (200 windows each); guard = both noise estimates agree")
rng = np.random.default_rng(SENS_SEED)
est_rows = []
for p, q, fixed_q in [(.02, .05, False), (.05, .05, False), (.10, .05, False), (.02, .15, False), (.02, .15, True)]:
    b = generate_streams_q(np.full(300, p), np.full(300, q), R, rng, 200)
    ev, mm, sp, _ = observables(b.observable, K.selfp)
    ph, qh, cov, ok = [], [], 0, 0
    for i in range(200):
        e = K.estimate(300, ev[i].mean(), ev[i].std(), mm[i].mean(), sp[i].mean(), sp[i].std(), q_fixed=Q0 if fixed_q else None)
        ph.append(e["p_hat_ev"]); qh.append(e["q_hat"])
        cov += e["p_lo_ev"] <= p <= e["p_hi_ev"]
        ok += not (e["p_hi_ev"] < e["p_lo_cf"] or e["p_hi_cf"] < e["p_lo_ev"])
    est_rows.append({"p": p, "q": q, "q_assumed_fixed": fixed_q, "p_hat_mean": float(np.mean(ph)), "p_hat_sd": float(np.std(ph)),
                     "q_hat_mean": float(np.mean(qh)), "p_ci_coverage": cov / 200, "guard_consistent": ok / 200})
    print(f"   true p={p:.2f} q={q:.2f} {'(q wrongly assumed 0.05)' if fixed_q else '':26s} p_hat {np.mean(ph):.4f}+-{np.std(ph):.4f}"
          f"  q_hat {np.mean(qh):.4f}  CI covers {cov / 200:.2f}  guard consistent {ok / 200:.2f}")

# ---------- B. measured inference runtime ----------
probe = np.random.default_rng(0).integers(0, 2, size=(100_000, 2 * R + 2)).astype(np.uint8)
runtime = {}
for k, d in decoders.items():
    t0 = perf_counter()
    d.predict_flip(probe)
    runtime[k] = 1e6 * (perf_counter() - t0) / len(probe)
print("\nB. MEASURED decoder inference time (microseconds per shot, this machine): " +
      ", ".join(f"{k} {v:.2f}" for k, v in runtime.items()))

# ---------- C. method comparison ----------
print(f"\nC. {TR} streams x {T} shots per scenario; costs in failure-equivalents; assumed costs: {costs}")
summary, raw, panels, ts_panels, ablation_rows = {}, {}, [], [], {m: {} for m in ABLATIONS}
overhead = {}
for (sname, (p_t, q_t, cs, ce)), child in zip(SCENARIOS.items(), np.random.SeedSequence(EVAL_SEED).spawn(len(SCENARIOS))):
    b = generate_streams_q(p_t, q_t, R, np.random.default_rng(child), TR)
    ev, mm, sp, codes = observables(b.observable, K.selfp)
    y = b.truth.y                                           # used only inside run_controller for counting failures
    runs = {}
    for m, spec in FIXED.items():
        runs[m] = [run_controller(FixedController(spec), ev[i], mm[i], sp[i], codes[i], y[i], K, costs, False) for i in range(TR)]
    for m, c in ADAPTIVE.items():
        runs[m] = [run_controller(AdaptiveController(K, c, Q0), ev[i], mm[i], sp[i], codes[i], y[i], K, costs, True) for i in range(TR)]
        overhead.setdefault(m, []).extend(1e6 * r["runtime_s"] / T for r in runs[m])
    pre = (0, T if cs is None else cs)
    post = None if cs is None else (ce, T)
    res = {}
    for m, rr in runs.items():
        cost = np.array([r["cost"] for r in rr])
        wrong = np.stack([r["wrong"] for r in rr])
        d = {"cost": mean_ci(cost), "failures_per_stream": float(wrong.sum(1).mean()),
             "switches": float(np.mean([r["n_switch"] for r in rr])), "updates": float(np.mean([r["n_update"] for r in rr]))}
        for wn, w in (("pre", pre), ("post", post)):
            if w:
                k_, n_ = int(wrong[:, w[0]:w[1]].sum()), TR * (w[1] - w[0])
                d[f"{wn}_failure"] = [k_ / n_, *wilson_interval(k_, n_)]
        logs = [r["log"] for r in rr]
        far = lambda l, t0: [e["t"] for e in l if e["event"] == "alarm" and (cs is None or e["t"] < t0)]
        d["false_alarms_per_stream"] = float(np.mean([len(far(l, cs or 0)) for l in logs]))
        d["unnecessary_actions_per_stream"] = float(np.mean([sum(1 for e in l if e["event"] == "decision" and e.get("why") == "alarm"
                                                                 and e["action"] in ("lookup", "minweight", "tree", "update")
                                                                 and (cs is None or e["t"] < cs)) for l in logs]))
        d["guard_vetoes_per_stream"] = float(np.mean([sum(1 for e in l if "guard veto" in str(e.get("action"))) for l in logs]))
        d["waits_per_stream"] = float(np.mean([sum(1 for e in l if e.get("action") == "wait") for l in logs]))
        if cs is not None:
            times = []
            for l in logs:
                acts = [e for e in l if e["event"] == "decision" and e["t"] >= cs and e["action"] in ("lookup", "minweight", "tree", "update")]
                if acts:
                    times.append(acts[0]["t"] + (costs.update_delay if acts[0]["action"] == "update" else 1) - cs)
            d["adaptation_time"] = {"median": float(np.median(times)) if times else None, "streams_with_action": len(times)}
        res[m] = d
        raw[f"{sname}|{m}|cost"] = cost
        raw[f"{sname}|{m}|bin_failures"] = wrong.reshape(TR, T // BIN, BIN).sum(2).astype(np.uint16)
    prop_cost = np.array([r["cost"] for r in runs["proposed"]])
    res_vs = {m: paired([r["cost"] for r in runs[m]], prop_cost) for m in runs if m != "proposed"}
    summary[sname] = {"methods": res, "cost_minus_proposed": res_vs}
    for m in ABLATIONS:
        v = res_vs[m]
        ablation_rows[m][sname] = (v["diff"], v["ci_low"], v["ci_high"])
    panels.append({"title": sname, "methods": {m: res[m]["cost"] for m in MAIN}})
    if cs is not None and sname.split()[0] in ("abrupt", "gradual", "q-shift") and sname in (
            "abrupt 0.02->0.10", "abrupt 0.10->0.02", "gradual 0.02->0.10", "q-shift 0.05->0.15 (p=0.05)"):
        curves = {}
        for m in ("fixed_minweight", "static_tree", "adaptive_baseline", "proposed"):
            cnt = raw[f"{sname}|{m}|bin_failures"].sum(0)
            ci = [wilson_interval(int(c), TR * BIN) for c in cnt]
            curves[m] = (cnt / (TR * BIN), np.array([c[0] for c in ci]), np.array([c[1] for c in ci]))
        ts_panels.append({"title": sname, "curves": curves, "change_start": cs, "change_end": ce, "median_alarm": {}})
    print(f"\n== {sname} ==  mean total cost per stream [95% CI] | failures | switches updates | false alarms, unnecessary actions, guard vetoes")
    for m in MAIN:
        d = res[m]
        c = d["cost"]
        print(f"  {m:18s} {c[0]:8.1f} [{c[1]:7.1f},{c[2]:7.1f}] | {d['failures_per_stream']:6.1f} | {d['switches']:.2f} {d['updates']:.2f} | "
              f"{d['false_alarms_per_stream']:.2f} {d['unnecessary_actions_per_stream']:.2f} {d['guard_vetoes_per_stream']:.2f}")
    best = min(runs, key=lambda m: np.mean([r["cost"] for r in runs[m]]))
    print(f"  cheapest on average: {best}")
    for m in ("fixed_minweight", "static_tree", "adaptive_baseline"):
        v = res_vs[m]
        print(f"  cost({m}) - cost(proposed): {v['diff']:+.1f} [{v['ci_low']:+.1f},{v['ci_high']:+.1f}]  Wilcoxon p={v['wilcoxon_p']:.3g}")

print("\nAblations (cost(method) - cost(proposed); positive = removing that part made things costlier)")
print(f"  {'method':18s} " + " ".join(f"{s[:17]:>17s}" for s in SCENARIOS))
for m in ABLATIONS:
    print(f"  {m:18s} " + " ".join(f"{ablation_rows[m][s][0]:>+17.1f}" for s in SCENARIOS))

# ---------- D. update-cost sensitivity (descriptive; NOT used to choose the headline) ----------
print("\nD. Update-cost sensitivity for the proposed controller (assumed update cost varies in both decision and accounting)")
sens = {}
for sname in ("abrupt 0.02->0.10", "stationary p=0.10"):
    p_t, q_t, cs, ce = SCENARIOS[sname]
    b = generate_streams_q(p_t, q_t, R, np.random.default_rng(np.random.SeedSequence(SENS_SEED + 1)), 30)
    ev, mm, sp, codes = observables(b.observable, K.selfp)
    y = b.truth.y
    print(f"  {sname}:  update_cost  updates/stream  cost - cost(static_tree) [95% CI]")
    for uc in (5, 10, 20, 40, 80, 160):
        cm = CostModel(update_cost=float(uc))
        base_c = np.array([run_controller(FixedController("tree"), ev[i], mm[i], sp[i], codes[i], y[i], K, cm, False)["cost"] for i in range(30)])
        pr = [run_controller(AdaptiveController(K, replace(P, costs=cm), Q0), ev[i], mm[i], sp[i], codes[i], y[i], K, cm, True) for i in range(30)]
        dv = paired([r["cost"] for r in pr], base_c)
        sens[f"{sname}|{uc}"] = {"updates": float(np.mean([r["n_update"] for r in pr])), **dv}
        print(f"      {uc:>10d} {np.mean([r['n_update'] for r in pr]):>14.2f}  {dv['diff']:+9.1f} [{dv['ci_low']:+.1f},{dv['ci_high']:+.1f}]")

print("\nE. MEASURED controller software overhead (microseconds per shot of Python control loop, this machine): " +
      ", ".join(f"{m} {np.mean(v):.2f}" for m, v in overhead.items()))

res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "exp11_controller_summary.json", "w") as f:
    json.dump({"config": {"trials": TR, "T": T, "eval_seed": EVAL_SEED, "sens_seed": SENS_SEED, "model_sha256": model_hashes,
                          "headline_controller": {k: v for k, v in P.__dict__.items() if k != "costs"},
                          "assumed_costs": costs.__dict__, "detector_calibration": "stage 8 (window z=3.5, 500/100 shots)"},
               "confidence_validation_exact": conf_table, "estimator_validation": est_rows,
               "measured_inference_us_per_shot": runtime,
               "measured_controller_overhead_us_per_shot": {m: float(np.mean(v)) for m, v in overhead.items()},
               "scenarios": summary, "update_cost_sensitivity": sens}, f, indent=2, default=float)
np.savez_compressed(res_dir / "exp11_controller_raw.npz", **{k.replace("|", "__"): v for k, v in raw.items()})
plot_method_costs(panels, fig_dir / "exp11_cost_by_scenario.png")
plot_ablation(ablation_rows, list(SCENARIOS), fig_dir / "exp11_ablation.png")
plot_failure_timeseries(ts_panels, BIN, fig_dir / "exp11_failure_timeseries.png", cols=2)
print(f"\nSaved results/exp11_controller_summary.json, exp11_controller_raw.npz and 3 figures in {fig_dir}")