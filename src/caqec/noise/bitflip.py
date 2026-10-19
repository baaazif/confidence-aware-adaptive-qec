from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, pauli_error


def build_bitflip_noise_model(p: float) -> NoiseModel:
    """Each 'id' gate on any qubit suffers an X error with probability p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    error = pauli_error([("X", p), ("I", 1.0 - p)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ["id"])
    return noise_model


def add_noise_slot(circuit: QuantumCircuit, qubits=None) -> QuantumCircuit:
    """Return a copy with one 'id' gate per chosen qubit (default: all qubits).

    These 'id' gates are the only places where the noise model acts.
    """
    if qubits is None:
        qubits = range(circuit.num_qubits)
    out = circuit.copy()
    out.barrier()
    for q in qubits:
        out.id(q)
    out.barrier()
    return out