from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from caqec.codes.repetition import build_logical_basis_circuit
from caqec.codes.syndrome import syndrome_extraction_circuit
from caqec.decoders.lookup import apply_correction, parse_syndrome
from caqec.noise.bitflip import add_noise_slot, build_bitflip_noise_model

# Protected circuit classical bits (5 total, one register):
#   clbit 0 = s1, clbit 1 = s2, clbit 2 = data q0, clbit 3 = data q1, clbit 4 = data q2.
# Qiskit prints clbit 4 on the LEFT, so a key looks like "d2 d1 d0 s2 s1" (no spaces).


def build_unprotected_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc = add_noise_slot(qc)  # one noisy 'id' on the single qubit
    qc.measure(0, 0)
    return qc


def build_protected_circuit(logical_bit: int = 0) -> QuantumCircuit:
    qc = QuantumCircuit(5, 5)
    qc.compose(build_logical_basis_circuit(logical_bit), qubits=[0, 1, 2], inplace=True)
    qc = add_noise_slot(qc, qubits=[0, 1, 2])  # noise on DATA qubits only
    qc.compose(syndrome_extraction_circuit(), clbits=[0, 1], inplace=True)
    qc.measure([0, 1, 2], [2, 3, 4])  # final readout of the data, used to score the shot
    return qc


def parse_shot(key: str):
    """'d2 d1 d0 s2 s1' -> (data bits (b0, b1, b2), syndrome (s1, s2))."""
    data = (int(key[2]), int(key[1]), int(key[0]))
    return data, parse_syndrome(key[3:5])


def logical_failure(data_bits, syndrome, logical_bit: int = 0) -> bool:
    """True if, after syndrome-based correction, the data is not the intended codeword."""
    corrected = apply_correction(data_bits, syndrome)
    return corrected != (logical_bit,) * 3


def _run(circuit, p, shots, seed, **sim_options):
    sim = AerSimulator(noise_model=build_bitflip_noise_model(p), **sim_options)
    tqc = transpile(circuit, sim, optimization_level=0)  # keep the 'id' noise gates
    return sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()


def run_unprotected(p: float, shots: int, seed: int, **sim_options) -> dict:
    counts = _run(build_unprotected_circuit(), p, shots, seed, **sim_options)
    return {"shots": shots, "failures": counts.get("1", 0)}


def run_protected(p: float, shots: int, seed: int, **sim_options) -> dict:
    counts = _run(build_protected_circuit(0), p, shots, seed, **sim_options)
    failures = any_flip = 0
    for key, n in counts.items():
        data, syndrome = parse_shot(key)
        if data != (0, 0, 0):
            any_flip += n  # a physical error happened (whether or not it was fixed)
        if logical_failure(data, syndrome):
            failures += n  # the decoder did not restore logical 0
    return {"shots": shots, "failures": failures, "any_flip": any_flip}