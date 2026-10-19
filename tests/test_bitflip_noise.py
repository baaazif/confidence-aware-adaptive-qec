import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from step04_noise_check import run_counts


def test_p0_never_flips():
    assert run_counts(0.0, shots=2000) == {"000": 2000}


def test_p1_always_flips_every_qubit():
    assert run_counts(1.0, shots=2000) == {"111": 2000}


def test_small_p_matches_theory_within_5_sigma():
    p, shots = 0.1, 50_000
    counts = run_counts(p, shots=shots, seed=7)
    sigma = (p * (1 - p) / shots) ** 0.5
    for q in range(3):
        freq = sum(n for s, n in counts.items() if s[2 - q] == "1") / shots
        assert abs(freq - p) < 5 * sigma


def test_invalid_p_rejected():
    from caqec.noise.bitflip import build_bitflip_noise_model
    with pytest.raises(ValueError):
        build_bitflip_noise_model(1.5)