"""Validation checks for the decoder dataset. Each check returns a Check; none train a model."""
from dataclasses import dataclass
from math import sqrt

import numpy as np

from caqec.decoders.lookup import apply_correction, compute_syndrome
from caqec.simulation.dataset import feature_names


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _within(observed, expected, n, z=5.0):
    sigma = sqrt(max(expected * (1 - expected), 0.0) / n)
    return abs(observed - expected) <= z * sigma + 1e-12, sigma


def check_schema(split, rounds, name):
    X, y = split["X"], split["y"]
    ok = (X.shape[1] == 2 * rounds + 2 and X.shape[0] == len(y)
          and set(np.unique(X)) <= {0, 1} and set(np.unique(y)) <= {0, 1}
          and X.dtype == np.uint8 and y.dtype == np.uint8)
    return Check(f"{name}: schema (binary features, {2 * rounds + 2} columns)", ok,
                 f"X{X.shape}, y{y.shape}")


def check_uid_disjoint(splits, names=("train", "val", "test")):
    seen, ok, detail = set(), True, []
    for n in names:
        uid = splits[n]["meta_trial_uid"]
        if len(np.unique(uid)) != len(uid):
            ok = False
            detail.append(f"{n} has duplicate trial ids")
        if seen & set(uid.tolist()):
            ok = False
            detail.append(f"{n} shares trial ids with an earlier split")
        seen |= set(uid.tolist())
    return Check("trial ids unique within and disjoint across splits", ok,
                 "; ".join(detail) or "no shared trial ids")


def check_independent_realizations(splits, names=("train", "val", "test"), threshold=0.9):
    """Same seed in two splits would make rows match position by position (fraction 1.0).
    Independent samples match far less often, even when many trials have no noise at all."""
    ok, parts = True, []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = splits[a]["meta_realization"], splits[b]["meta_realization"]
            n = min(len(A), len(B))
            frac = float((A[:n] == B[:n]).all(axis=1).mean())
            parts.append(f"{a}/{b}={frac:.3f}")
            if frac > threshold:
                ok = False
    return Check("splits come from independent random streams (position-wise match fraction)",
                 ok, ", ".join(parts) + f"  (fail if > {threshold})")


def check_no_truth_in_features(split, rounds, name):
    X, y, e = split["X"], split["y"], split["meta_e_final"]
    names = feature_names(rounds)
    bad_names = [n for n in names if n.startswith("meta") or n.startswith("e_")]
    copies = [q for q in range(3) if any(np.array_equal(X[:, c], e[:, q]) for c in range(X.shape[1]))]
    corrs = [abs(np.corrcoef(X[:, c], y)[0, 1]) if X[:, c].std() > 0 else 0.0
             for c in range(X.shape[1])]
    ok = not bad_names and not copies and max(corrs) < 0.9
    return Check(f"{name}: features contain no ground-truth columns", ok,
                 f"max |corr(feature, y)| = {max(corrs):.3f}")


def _scalar_row(row, rounds):
    """Independent re-computation using the scalar baseline functions."""
    e, measured = [0, 0, 0], []
    for r in range(rounds):
        for q in range(3):
            e[q] ^= int(row[r * 3 + q])
        s = compute_syndrome(e)
        f = row[3 * rounds + 2 * r: 3 * rounds + 2 * r + 2]
        measured += [s[0] ^ int(f[0]), s[1] ^ int(f[1])]
    s_final = compute_syndrome(e)
    measured += list(s_final)
    corrected = apply_correction(tuple(e), s_final)
    return measured, tuple(e), int(corrected != (0, 0, 0))


def check_labels_recomputed(split, rounds, name, n_sample=3000, seed=0):
    idx = np.random.default_rng(seed).choice(len(split["y"]), min(n_sample, len(split["y"])),
                                             replace=False)
    bad = 0
    for i in idx:
        measured, e, y = _scalar_row(split["meta_realization"][i], rounds)
        if (measured != split["X"][i].tolist() or e != tuple(split["meta_e_final"][i].tolist())
                or y != int(split["y"][i])):
            bad += 1
    return Check(f"{name}: stored rows match independent scalar recomputation", bad == 0,
                 f"{bad} mismatches in {len(idx)} sampled rows")


def check_noise_rates(split, cfg, name):
    R, real = cfg.rounds, split["meta_realization"]
    ok, parts = True, []
    for p in cfg.p_values:
        m = np.isclose(split["meta_p"], p)
        rate = real[m, : 3 * R].mean()
        good, sigma = _within(rate, p, m.sum() * 3 * R)
        ok &= good
        parts.append(f"p={p}: {rate:.4f}")
    meas = real[:, 3 * R:].mean()
    good, _ = _within(meas, cfg.q, real.shape[0] * 2 * R)
    ok &= good
    return Check(f"{name}: empirical data-flip and measurement-flip rates match config", ok,
                 ", ".join(parts) + f", q: {meas:.4f} (config {cfg.q})")


def p_effective(p, rounds):
    """Chance a data qubit has flipped an odd number of times after `rounds` rounds."""
    return (1 - (1 - 2 * p) ** rounds) / 2


def check_label_vs_theory(split, cfg, name):
    ok, parts = True, []
    for p in cfg.p_values:
        m = np.isclose(split["meta_p"], p)
        pe = p_effective(p, cfg.rounds)
        theory = 3 * pe**2 - 2 * pe**3
        obs = split["y"][m].mean()
        good, _ = _within(obs, theory, m.sum())
        ok &= good
        parts.append(f"p={p}: {obs:.4f} vs {theory:.4f}")
    return Check(f"{name}: P(y=1) matches 3p_eff^2 - 2p_eff^3", ok, "; ".join(parts))


def check_calibration_set(cal):
    """rounds=1, q=0, p=0.1: ties the generator to the baseline experiment's formulas."""
    p, y, X = 0.1, cal["y"], cal["X"]
    n = len(y)
    results, ok = [], True
    obs = y.mean()
    theory = 3 * p**2 - 2 * p**3
    g, _ = _within(obs, theory, n); ok &= g
    results.append(f"P(y=1) {obs:.5f} vs {theory:.5f}")
    nontrivial = X[:, -2:].any(axis=1)
    obs = y[nontrivial].mean()
    g, _ = _within(obs, p, int(nontrivial.sum())); ok &= g
    results.append(f"P(y=1 | non-trivial syndrome) {obs:.4f} vs p={p}")
    obs = y[~nontrivial].mean()
    exp = p**3 / ((1 - p) ** 3 + p**3)
    g, _ = _within(obs, exp, int((~nontrivial).sum())); ok &= g
    results.append(f"P(y=1 | syndrome 00) {obs:.5f} vs {exp:.5f}")
    return Check("calibration set (R=1, q=0, p=0.1) agrees with the baseline theory", ok,
                 "; ".join(results))


def _codes(X):
    return X @ (1 << np.arange(X.shape[1]))


def ambiguity_audit(train, rounds):
    """Descriptive only: how many identical inputs carry different labels (irreducible overlap)."""
    X, y = train["X"], train["y"]
    size = 2 ** X.shape[1]
    code = _codes(X)
    n0 = np.bincount(code[y == 0], minlength=size)
    n1 = np.bincount(code[y == 1], minlength=size)
    both = (n0 > 0) & (n1 > 0)
    conflict_rows = int((n0 + n1)[both].sum()) / len(y)
    floor = float(np.minimum(n0, n1).sum()) / len(y)
    return Check("ambiguity audit (train, informational)", True,
                 f"{int(((n0 + n1) > 0).sum())} distinct input patterns of {size} possible; "
                 f"{conflict_rows:.3f} of rows share their input with a row of the other label; "
                 f"lowest error ANY classifier could reach on these training inputs = {floor:.4f} "
                 "(optimistic: input patterns seen only once contribute 0 by construction)")


def unseen_pattern_fraction(train, other, name):
    seen = np.bincount(_codes(train["X"]), minlength=2 ** train["X"].shape[1]) > 0
    frac = float((~seen[_codes(other["X"])]).mean())
    return Check(f"{name}: input patterns never seen in train (informational)", True,
                 f"{frac:.4f} of rows. Identical inputs across splits are expected "
                 "(few possible patterns) and are not leakage; shared trials would be.")


def run_all_checks(splits, cfg):
    out = [check_uid_disjoint(splits), check_independent_realizations(splits)]
    for name in ("train", "val", "test"):
        s = splits[name]
        out += [check_schema(s, cfg.rounds, name),
                check_no_truth_in_features(s, cfg.rounds, name),
                check_labels_recomputed(s, cfg.rounds, name),
                check_noise_rates(s, cfg, name),
                check_label_vs_theory(s, cfg, name)]
    out.append(check_schema(splits["calibration"], 1, "calibration"))
    out.append(check_calibration_set(splits["calibration"]))
    out.append(ambiguity_audit(splits["train"], cfg.rounds))
    out += [unseen_pattern_fraction(splits["train"], splits[n], n) for n in ("val", "test")]
    return out