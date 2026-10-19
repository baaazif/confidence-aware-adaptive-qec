from itertools import product

import numpy as np

from caqec.decoders.lookup import apply_correction, compute_syndrome
from caqec.evaluation.dataset_checks import (_scalar_row, check_independent_realizations,
                                             check_uid_disjoint, run_all_checks)
from caqec.simulation.dataset import (CORRECTION_TABLE, DatasetConfig, generate_all,
                                      generate_split, rows_from_realization)

SMALL = DatasetConfig(n_train=4000, n_val=1000, n_test=1000)


def test_correction_table_matches_baseline_decoder():
    for s1, s2 in product((0, 1), repeat=2):
        expected = apply_correction((0, 0, 0), (s1, s2))
        assert tuple(CORRECTION_TABLE[2 * s1 + s2]) == expected


def test_label_rule_on_all_single_round_patterns():
    rows = np.array([list(p) + [0, 0] for p in product((0, 1), repeat=3)], dtype=np.uint8)
    X, e, y = rows_from_realization(rows, rounds=1)
    for i, pattern in enumerate(product((0, 1), repeat=3)):
        assert y[i] == (sum(pattern) >= 2)               # same rule as the baseline decoder
        assert tuple(X[i, -2:]) == compute_syndrome(pattern)
        assert _scalar_row(rows[i], 1)[2] == y[i]


def test_ambiguous_syndrome_gets_both_labels():
    # 100 and 011 share syndrome (1,0) but get different labels: never relabeled
    rows = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    X, _, y = rows_from_realization(rows, 1)
    assert tuple(X[0, -2:]) == tuple(X[1, -2:]) == (1, 0)
    assert list(y) == [0, 1]


def test_degenerate_histories_share_a_label():
    # 3 rounds, qubit 0 flips in rounds 1 and 3 (net 0)  vs  no flips at all: same net error
    a = np.zeros((1, 15), dtype=np.uint8); a[0, 0] = 1; a[0, 6] = 1
    b = np.zeros((1, 15), dtype=np.uint8)
    _, ea, ya = rows_from_realization(a, 3)
    _, eb, yb = rows_from_realization(b, 3)
    assert (ea == eb).all() and ya[0] == yb[0] == 0


def test_measurement_errors_never_change_the_label():
    clean = np.zeros((1, 5), dtype=np.uint8); clean[0, 0] = 1
    noisy = clean.copy(); noisy[0, 3] = 1  # flip a syndrome readout only
    Xc, _, yc = rows_from_realization(clean, 1)
    Xn, _, yn = rows_from_realization(noisy, 1)
    assert yc[0] == yn[0]
    assert Xc[0, 0] != Xn[0, 0]                    # the measured feature did change


def test_boundaries_p0_and_p1():
    seed = np.random.SeedSequence(1)
    zero = generate_split(DatasetConfig(p_values=(0.0,), q=0.0), "train", 500, seed)
    assert zero["X"].sum() == 0 and zero["y"].sum() == 0
    one_round = generate_split(DatasetConfig(rounds=1, p_values=(1.0,), q=0.0), "train", 500, seed)
    assert (one_round["y"] == 1).all()             # 111 -> syndrome 00 -> uncorrected -> failure
    two_rounds = generate_split(DatasetConfig(rounds=2, p_values=(1.0,), q=0.0), "train", 500, seed)
    assert (two_rounds["y"] == 0).all()            # flipped twice -> back to 000


def test_same_seed_reproducible_and_different_seed_differs():
    a = generate_split(SMALL, "train", 300, np.random.SeedSequence(5))
    b = generate_split(SMALL, "train", 300, np.random.SeedSequence(5))
    c = generate_split(SMALL, "train", 300, np.random.SeedSequence(6))
    assert np.array_equal(a["meta_realization"], b["meta_realization"])
    assert not np.array_equal(a["meta_realization"], c["meta_realization"])


def test_leakage_checks_catch_a_shared_seed():
    # a fresh SeedSequence(9) each time = genuinely the same seed (spawn() is stateful,
    # so reusing ONE object would silently give different streams)
    same = {name: generate_split(SMALL, name, 500, np.random.SeedSequence(9))
            for name in ("train", "val", "test")}
    assert not check_independent_realizations(same).ok
    leaked = {k: dict(v) for k, v in same.items()}
    leaked["val"]["meta_trial_uid"] = leaked["train"]["meta_trial_uid"]
    assert not check_uid_disjoint(leaked).ok


def test_full_validation_suite_passes_on_a_proper_dataset():
    checks = run_all_checks(generate_all(SMALL), SMALL)
    failed = [c.name + ": " + c.detail for c in checks if not c.ok]
    assert not failed, failed