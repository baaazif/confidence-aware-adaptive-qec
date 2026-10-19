from itertools import product

import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

from caqec.codes.syndrome import circuit_with_explicit_errors
from caqec.decoders.lookup import (apply_correction, compute_syndrome,
                                   decode_syndrome, majority_vote, parse_syndrome)

SIM = AerSimulator()


def circuit_syndrome(logical_bit, error_qubits):
    qc = circuit_with_explicit_errors(logical_bit, error_qubits)
    counts = SIM.run(transpile(qc, SIM), shots=50).result().get_counts()
    assert len(counts) == 1, f"syndrome should be deterministic, got {counts}"
    return parse_syndrome(next(iter(counts)))


@pytest.mark.parametrize("logical_bit", [0, 1])
@pytest.mark.parametrize("errors,expected", [
    ((), (0, 0)), ((0,), (1, 0)), ((1,), (1, 1)), ((2,), (0, 1)),
])
def test_single_error_syndromes(logical_bit, errors, expected):
    assert circuit_syndrome(logical_bit, errors) == expected


@pytest.mark.parametrize("logical_bit", [0, 1])
def test_circuit_matches_classical_syndrome_for_all_patterns(logical_bit):
    for pattern in product((0, 1), repeat=3):
        qubits = [q for q, b in enumerate(pattern) if b]
        assert circuit_syndrome(logical_bit, qubits) == compute_syndrome(pattern)


def test_decoder_fixes_each_single_flip():
    for q in range(3):
        bits = [0, 0, 0]
        bits[q] = 1
        assert apply_correction(tuple(bits), compute_syndrome(bits)) == (0, 0, 0)


def test_decoder_picks_the_flipped_qubit():
    assert decode_syndrome((1, 0)) == 0
    assert decode_syndrome((1, 1)) == 1
    assert decode_syndrome((0, 1)) == 2
    assert decode_syndrome((0, 0)) is None


def test_two_flips_are_miscorrected_and_syndrome_is_ambiguous():
    # X on q0 and q1 gives the same syndrome as a single X on q2 ...
    assert compute_syndrome((1, 1, 0)) == compute_syndrome((0, 0, 1))
    # ... so the decoder flips q2 and lands on 111: a logical error, undetected.
    assert apply_correction((1, 1, 0), compute_syndrome((1, 1, 0))) == (1, 1, 1)


def test_syndrome_decoder_and_majority_vote_agree_on_logical_outcome():
    for pattern in product((0, 1), repeat=3):
        corrected = apply_correction(pattern, compute_syndrome(pattern))
        assert corrected in {(0, 0, 0), (1, 1, 1)}
        assert corrected[0] == majority_vote(pattern)