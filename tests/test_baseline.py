from itertools import product

import pytest

from caqec.decoders.lookup import compute_syndrome
from caqec.evaluation.stats import wilson_interval
from caqec.evaluation.theory import (any_physical_flip, protected_failure,
                                     unprotected_failure)
from caqec.simulation.baseline import (logical_failure, parse_shot,
                                       run_protected, run_unprotected)


def test_parse_shot_known_keys():
    # X on q0: data (1,0,0) -> clbit2=1 -> "001"; syndrome (s1,s2)=(1,0) -> "01"
    assert parse_shot("00101") == ((1, 0, 0), (1, 0))
    # X on q2: data (0,0,1) -> "100"; syndrome (0,1) -> "10"
    assert parse_shot("10010") == ((0, 0, 1), (0, 1))


def test_logical_failure_iff_two_or_more_flips():
    for pattern in product((0, 1), repeat=3):
        failed = logical_failure(pattern, compute_syndrome(pattern))
        assert failed == (sum(pattern) >= 2)


def test_noise_free_has_no_failures_and_no_flips():
    assert run_unprotected(0.0, 2000, 1)["failures"] == 0
    r = run_protected(0.0, 2000, 1)
    assert r["failures"] == 0 and r["any_flip"] == 0


def test_p_equals_one_boundary():
    assert run_unprotected(1.0, 1000, 1)["failures"] == 1000
    r = run_protected(1.0, 1000, 1)  # all three flip -> 111 -> syndrome (0,0) -> uncorrected
    assert r["failures"] == 1000 and r["any_flip"] == 1000


def test_theory_boundaries():
    assert protected_failure(0.0) == 0 and protected_failure(1.0) == 1
    assert protected_failure(0.5) == pytest.approx(0.5)
    assert protected_failure(0.1) == pytest.approx(0.028)
    assert unprotected_failure(0.3) == 0.3
    assert any_physical_flip(0.1) == pytest.approx(1 - 0.9**3)


def test_wilson_known_values_and_edges():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(0, 1000)
    assert lo == 0.0 and 0 < hi < 0.01
    lo, hi = wilson_interval(1000, 1000)
    assert hi == 1.0 and lo > 0.99


def test_protected_matches_theory_within_5_sigma():
    p, n = 0.1, 50_000
    r = run_protected(p, n, seed=11)
    pl = protected_failure(p)
    sigma = (pl * (1 - pl) / n) ** 0.5
    assert abs(r["failures"] / n - pl) < 5 * sigma


def test_code_helps_below_half_and_hurts_above():
    n = 20_000
    assert run_protected(0.1, n, 3)["failures"] < run_unprotected(0.1, n, 4)["failures"]
    assert run_protected(0.7, n, 3)["failures"] > run_unprotected(0.7, n, 4)["failures"]


def test_same_seed_gives_same_result():
    assert run_protected(0.2, 5000, 99) == run_protected(0.2, 5000, 99)