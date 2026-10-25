"""Stage 8: non-stationary noise, static decoders before/after a change, and online change detection.

Design-time threshold calibration uses SEPARATE stationary streams (seed family 20261030).
Evaluation uses fresh streams (20261031). Detectors receive only the per-shot detection-event count of
the observable syndromes; labels, change times and noise levels are used only for scoring and plots.
Opens no dataset files; the test split is not touched."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from caqec.decoders.base import pattern_codes
from caqec.decoders.baseline import LookupDecoder
from caqec.decoders.exact import ExactDecoder
from caqec.decoders.minweight import MinWeightDecoder
from caqec.detection.online import CusumDetector, WindowDetector, defect_counts, run_detector
from caqec.evaluation.decoder_metrics import paired_difference
from caqec.evaluation.detection_metrics import score_detection, summarize_detection
from caqec.evaluation.frozen import load_frozen_models
from caqec.evaluation.stats import wilson_interval
from caqec.plotting.nonstationary_plots import plot_detection_summary, plot_failure_timeseries
from caqec.simulation.dataset import DatasetConfig
from caqec.simulation.nonstationary import NoiseSchedule, generate_streams

CAL_SEED, EVAL_SEED = 20261030, 20261031
T, WARM, BIN, TRIALS, MAX_DELAY = 6000, 500, 250, 100, 2000
CAL_STREAMS, CAL_P = 60, (0.02, 0.05, 0.10)
FA_TARGET_DEFAULT = 0.1                           # false alarms per 1000 shots (0.1 = about 0.5 per 5500-shot stream)
K, W_CUR = 0.5, 100
H_GRID = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20)
Z_GRID = (2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 7, 8)
SCENARIOS = {
    "stationary p=0.02": NoiseSchedule("stationary", 0.02, 0.02),
    "stationary p=0.05": NoiseSchedule("stationary", 0.05, 0.05),
    "abrupt 0.02->0.03": NoiseSchedule("abrupt", 0.02, 0.03, change_time=3000),
    "abrupt 0.02->0.05": NoiseSchedule("abrupt", 0.02, 0.05, change_time=3000),
    "abrupt 0.02->0.10": NoiseSchedule("abrupt", 0.02, 0.10, change_time=3000),
    "abrupt 0.05->0.10": NoiseSchedule("abrupt", 0.05, 0.10, change_time=3000),
    "abrupt 0.10->0.05": NoiseSchedule("abrupt", 0.10, 0.05, change_time=3000),
    "gradual 0.02->0.10": NoiseSchedule("gradual", 0.02, 0.10, change_time=2500, ramp_len=2000),
}

ap = argparse.ArgumentParser()
ap.add_argument("--models-dir", default=str(ROOT / "models"))
ap.add_argument("--exp04-results", default=str(ROOT / "results" / "exp04_stage2_selection_val.json"))
ap.add_argument("--results-dir", default=str(ROOT / "results"))
ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
ap.add_argument("--trials", type=int, default=TRIALS)
ap.add_argument("--fa-target", type=float, default=FA_TARGET_DEFAULT, help="max false alarms per 1000 shots at calibration")
args = ap.parse_args()
TR, FA_TARGET = args.trials, args.fa_target

cfg = DatasetConfig()
Q, R = cfg.q, cfg.rounds
models, model_hashes = load_frozen_models(args.models_dir, args.exp04_results)
decoders = {"lookup": LookupDecoder(), "minweight": MinWeightDecoder(R), "tree": models["tree"], "logreg": models["logreg"]}
ALL_X = ((np.arange(2 ** (2 * R + 2))[:, None] >> np.arange(2 * R + 2)) & 1).astype(np.uint8)
# every decoder is a pure function of the 8 syndrome bits, so its decision for all 256 patterns is a table
flip_tables = {k: d.predict_flip(ALL_X) for k, d in decoders.items()}
_exact = {}


def exact_at(p):
    if p not in _exact:
        _exact[p] = ExactDecoder(replace(cfg, p_values=(p,)))
    return _exact[p]


def analytic(name, p):
    return exact_at(p).expected_failure_rate(flip_tables[name].astype(int))


def aware_oracle(p):
    return exact_at(p).expected_failure_rate()


# ---------------- 1. design-time threshold calibration (stationary streams only) ----------------
print(f"Calibration: {CAL_STREAMS} stationary streams x {T} shots at each p in {CAL_P}; target <= {FA_TARGET} false alarms per 1000 shots\n")
cal_stats = {}
for p, child in zip(CAL_P, np.random.SeedSequence(CAL_SEED).spawn(len(CAL_P))):
    b = generate_streams(NoiseSchedule("stationary", p, p), T, Q, R, np.random.default_rng(child), CAL_STREAMS)
    cal_stats[p] = defect_counts(b.observable)


def fa_rate(make, p):
    n = sum(len(run_detector(make(), s)) for s in cal_stats[p])
    return 1000 * n / (CAL_STREAMS * (T - WARM))


calibration = {"cusum": {}, "window": {}}
for name, grid, mk in (("cusum", H_GRID, lambda v: (lambda: CusumDetector(WARM, K, v))),
                       ("window", Z_GRID, lambda v: (lambda: WindowDetector(WARM, W_CUR, v)))):
    print(f"{name:7s} " + " ".join(f"{('h' if name == 'cusum' else 'z')}={v:<5}" for v in grid))
    for p in CAL_P:
        rates = [fa_rate(mk(v), p) for v in grid]
        calibration[name][str(p)] = dict(zip(map(str, grid), rates))
        print(f"  p={p:<5}" + " ".join(f"{r:7.3f}" for r in rates))
chosen = {}
for name, grid in (("cusum", H_GRID), ("window", Z_GRID)):
    ok = [v for v in grid if max(calibration[name][str(p)][str(v)] for p in CAL_P) <= FA_TARGET]
    chosen[name] = ok[0] if ok else grid[-1]
    if not ok:
        print(f"WARNING: no {name} setting met the target; using the largest ({grid[-1]})")
print(f"\nChosen: CUSUM h={chosen['cusum']} (k={K}), window z={chosen['window']} (w_ref={WARM}, w_cur={W_CUR}); warm-up {WARM} shots\n")
make_det = {"cusum": lambda: CusumDetector(WARM, K, chosen["cusum"]), "window": lambda: WindowDetector(WARM, W_CUR, chosen["window"])}

# ---------------- 2. evaluation ----------------
summary, panels, det_rows = {}, [], []
raw = {"bin_size": np.array(BIN), "T": np.array(T)}
print(f"Evaluation: {TR} streams x {T} shots per scenario; detection counts as detected if first alarm is within {MAX_DELAY} shots of the change\n")
for (label, sch), child in zip(SCENARIOS.items(), np.random.SeedSequence(EVAL_SEED).spawn(len(SCENARIOS))):
    batch = generate_streams(sch, T, Q, R, np.random.default_rng(child), TR)
    X, y = batch.observable, batch.truth.y                        # y is used ONLY for scoring below
    stats = defect_counts(X)                                      # the only thing detectors receive
    codes = pattern_codes(X.reshape(-1, X.shape[-1])).reshape(TR, T)
    preds = {k: flip_tables[k][codes] for k in decoders}
    wrong = {k: preds[k] != y for k in decoders}
    cs, ce = sch.change_start, sch.change_end
    alarms = {d: [run_detector(make_det[d](), stats[i]) for i in range(TR)] for d in make_det}
    det_summary = {d: summarize_detection([score_detection(a, T, WARM, cs, MAX_DELAY) for a in alarms[d]]) for d in alarms}
    windows = {"pre": (0, T if cs is None else cs)}
    if cs is not None:
        windows["post"] = (ce, T)
    dec = {}
    for wname, (lo, hi) in windows.items():
        p_here = sch.p0 if wname == "pre" else sch.p1
        dec[wname] = {"p": p_here, "aware_oracle_analytic": aware_oracle(p_here), "decoders": {}}
        for k in decoders:
            kfail, n = int(wrong[k][:, lo:hi].sum()), TR * (hi - lo)
            ci = wilson_interval(kfail, n)
            g = paired_difference(y[:, lo:hi].ravel(), preds["lookup"][:, lo:hi].ravel(), preds[k][:, lo:hi].ravel()) if k != "lookup" else None
            dec[wname]["decoders"][k] = {"failure_rate": kfail / n, "ci": list(ci), "analytic": analytic(k, p_here),
                                         "gain_vs_lookup": None if g is None else {kk: g[kk] for kk in ("diff", "ci_low", "ci_high")}}
    summary[label] = {"schedule": sch.__dict__ | {"steps": list(sch.steps)}, "detection": det_summary, "decoders": dec}
    # ---- print ----
    print(f"== {label} ==")
    for d, s in det_summary.items():
        base = f"  {d:6s} false alarms {s['false_alarms_per_1000_shots']:.3f}/1000 shots (any in {s['p_any_false_alarm']:.2f} of streams)"
        if "p_detect" in s:
            dm = "n/a" if s["delay_median"] is None else f"median {s['delay_median']:.0f}, mean {s['delay_mean']:.0f}, p90 {s['delay_p90']:.0f}"
            base += f" | detected {s['p_detect']:.2f} [{s['p_detect_ci'][0]:.2f},{s['p_detect_ci'][1]:.2f}], missed {s['missed']}, delay {dm}"
        print(base)
    for wname, w in dec.items():
        cells = " ".join(f"{k}: {v['failure_rate']:.4f}" for k, v in w["decoders"].items())
        print(f"  {wname:4s} (p={w['p']}) failure  {cells} | aware-oracle analytic {w['aware_oracle_analytic']:.4f}")
    # ---- raw data + plot inputs ----
    counts = np.stack([wrong[k].reshape(TR, T // BIN, BIN).sum(2) for k in decoders], axis=1).astype(np.uint16)
    key = label.replace(" ", "_").replace(">", "")
    raw[f"{key}__bin_failures[trial,decoder,bin]"] = counts
    for d in alarms:
        width = max((len(a) for a in alarms[d]), default=0)
        pad = -np.ones((TR, max(width, 1)), dtype=np.int32)
        for i, a in enumerate(alarms[d]):
            pad[i, :len(a)] = a
        raw[f"{key}__{d}_alarms"] = pad
    pooled = counts.sum(axis=0)
    curves = {}
    for j, k in enumerate(decoders):
        ci = [wilson_interval(int(c), TR * BIN) for c in pooled[j]]
        curves[k] = (pooled[j] / (TR * BIN), np.array([c[0] for c in ci]), np.array([c[1] for c in ci]))
    med = {}
    for d in alarms:
        first = [next((a for a in al if a >= (cs or 0)), None) for al in alarms[d]]
        first = [f for f in first if f is not None]
        med[d] = float(np.median(first)) if (cs is not None and first) else None
    panels.append({"title": label, "curves": curves, "change_start": cs, "change_end": ce, "median_alarm": med})
    if cs is not None:
        det_rows.append({"label": label, "cusum": det_summary["cusum"], "window": det_summary["window"]})

res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
res_dir.mkdir(parents=True, exist_ok=True)
config = {"T": T, "warmup": WARM, "bin": BIN, "trials": TR, "max_delay": MAX_DELAY, "fa_target_per_1000": FA_TARGET,
          "cusum_k": K, "window_w_cur": W_CUR, "chosen": chosen, "cal_seed": CAL_SEED, "eval_seed": EVAL_SEED, "q": Q,
          "model_sha256": model_hashes, "numpy": np.__version__}
with open(res_dir / "exp10_nonstationary_summary.json", "w") as f:
    json.dump({"config": config, "calibration_fa_per_1000": calibration, "scenarios": summary}, f, indent=2)
np.savez_compressed(res_dir / "exp10_nonstationary_raw.npz", **raw)
plot_failure_timeseries(panels, BIN, fig_dir / "exp10_failure_timeseries.png")
plot_detection_summary(det_rows, fig_dir / "exp10_detection_summary.png")
print(f"\nSaved results/exp10_nonstationary_summary.json, exp10_nonstationary_raw.npz and 2 figures in {fig_dir}")