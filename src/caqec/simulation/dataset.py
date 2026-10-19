"""Dataset generation for a future ML decoder (3-qubit repetition code).

Noise model (phenomenological, NumPy only):
  for each of `rounds` rounds: every data qubit flips with probability p (flips accumulate),
  then both syndrome bits are measured, each readout wrong with probability q.
  After the last round the data qubits are read out perfectly, which gives one final,
  error-free syndrome ("s_final").

Features X  : measured syndromes of every round, then s_final.   Nothing else.
Label y     : 1 if the baseline lookup correction for s_final leaves a residual
              logical X (XXX), else 0.  See docs/dataset_spec.md for the full rules.
Everything prefixed `meta_` is for auditing only and must never be used as a feature.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from caqec.decoders.lookup import SYNDROME_TO_FLIP

SPLIT_OFFSETS = {"train": 0, "val": 10_000_000, "test": 20_000_000, "calibration": 30_000_000}


@dataclass(frozen=True)
class DatasetConfig:
    rounds: int = 3
    p_values: tuple = (0.02, 0.05, 0.10)
    q: float = 0.05
    n_train: int = 20_000  # trials per p value
    n_val: int = 5_000
    n_test: int = 5_000
    master_seed: int = 20261019


def _build_correction_table() -> np.ndarray:
    """Baseline decoder as an array: row index = 2*s1 + s2, columns = qubits to flip."""
    table = np.zeros((4, 3), dtype=np.uint8)
    for (s1, s2), flip in SYNDROME_TO_FLIP.items():
        if flip is not None:
            table[2 * s1 + s2, flip] = 1
    return table


CORRECTION_TABLE = _build_correction_table()


def syndrome_from_errors(e: np.ndarray) -> np.ndarray:
    """(..., 3) error bits -> (..., 2) syndrome bits (e0 xor e1, e1 xor e2)."""
    return np.stack([e[..., 0] ^ e[..., 1], e[..., 1] ^ e[..., 2]], axis=-1)


def reference_correction(s_final: np.ndarray) -> np.ndarray:
    return CORRECTION_TABLE[2 * s_final[:, 0] + s_final[:, 1]]


def logical_flip_label(e_final: np.ndarray, s_final: np.ndarray) -> np.ndarray:
    """Rule 1: y = 1 iff (true error XOR baseline correction) = 111."""
    residual = e_final ^ reference_correction(s_final)
    weight = residual.sum(axis=1)
    if not np.all((weight == 0) | (weight == 3)):
        raise ValueError("residual is not a codeword: s_final does not match e_final")
    return (weight == 3).astype(np.uint8)


def feature_names(rounds: int) -> list:
    names = [f"s{k}_r{r}" for r in range(1, rounds + 1) for k in (1, 2)]
    return names + ["s1_final", "s2_final"]


def rows_from_realization(realization: np.ndarray, rounds: int):
    """Pure function: stored noise draws -> (X, e_final, y).

    realization layout per trial: 3*rounds data-flip bits (round-major, qubit-minor)
    followed by 2*rounds measurement-flip bits (round-major, syndrome-bit-minor).
    """
    n = realization.shape[0]
    data_flips = realization[:, : 3 * rounds].reshape(n, rounds, 3)
    meas_flips = realization[:, 3 * rounds:].reshape(n, rounds, 2)
    e_hist = np.bitwise_xor.accumulate(data_flips, axis=1)  # net error after each round
    measured = syndrome_from_errors(e_hist) ^ meas_flips
    e_final = e_hist[:, -1, :]
    s_final = syndrome_from_errors(e_final)
    X = np.concatenate([measured.reshape(n, 2 * rounds), s_final], axis=1)
    return X, e_final, logical_flip_label(e_final, s_final)


def generate_split(cfg: DatasetConfig, split_name: str, n_per_p: int, seed_seq) -> dict:
    """One split, generated from its OWN seed sequence (independent of other splits)."""
    children = seed_seq.spawn(len(cfg.p_values) + 1)  # one stream per p, one for shuffling
    realizations, p_col = [], []
    for p, child in zip(cfg.p_values, children[:-1]):
        rng = np.random.default_rng(child)
        data = (rng.random((n_per_p, cfg.rounds, 3)) < p).astype(np.uint8)
        meas = (rng.random((n_per_p, cfg.rounds, 2)) < cfg.q).astype(np.uint8)
        realizations.append(np.concatenate([data.reshape(n_per_p, -1),
                                            meas.reshape(n_per_p, -1)], axis=1))
        p_col.append(np.full(n_per_p, p))
    realization = np.concatenate(realizations)
    p_col = np.concatenate(p_col)
    uid = SPLIT_OFFSETS[split_name] + np.arange(len(realization), dtype=np.int64)
    perm = np.random.default_rng(children[-1]).permutation(len(realization))
    realization, p_col, uid = realization[perm], p_col[perm], uid[perm]
    X, e_final, y = rows_from_realization(realization, cfg.rounds)
    return {"X": X, "y": y, "meta_p": p_col, "meta_q": np.full(len(y), cfg.q),
            "meta_trial_uid": uid, "meta_e_final": e_final, "meta_realization": realization}


def generate_all(cfg: DatasetConfig) -> dict:
    """train / val / test from three independent seed streams, plus a calibration set
    (rounds=1, q=0, p=0.1) used only to tie the generator back to the baseline experiment."""
    train_s, val_s, test_s, cal_s = np.random.SeedSequence(cfg.master_seed).spawn(4)
    cal_cfg = replace(cfg, rounds=1, q=0.0, p_values=(0.1,))
    return {
        "train": generate_split(cfg, "train", cfg.n_train, train_s),
        "val": generate_split(cfg, "val", cfg.n_val, val_s),
        "test": generate_split(cfg, "test", cfg.n_test, test_s),
        "calibration": generate_split(cal_cfg, "calibration", 200_000, cal_s),
    }


def checksum(arrays: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(np.ascontiguousarray(arrays[key]).tobytes())
    return h.hexdigest()


def save_split(path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)


def load_split(path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def write_card(path, cfg: DatasetConfig, splits: dict) -> None:
    card = {
        "config": asdict(cfg),
        "feature_names": feature_names(cfg.rounds),
        "label_rule": "y=1 iff true net data error XOR baseline lookup correction(s_final) == 111",
        "numpy_version": np.__version__,
        "sha256": {name: checksum(arrays) for name, arrays in splits.items()},
        "rows": {name: int(len(arrays["y"])) for name, arrays in splits.items()},
    }
    with open(path, "w") as f:
        json.dump(card, f, indent=2)