from qiskit import QuantumCircuit

from caqec.codes.repetition import build_logical_basis_circuit

# Qubit layout: data qubits 0,1,2; ancilla qubits 3,4.
# Classical bits: clbit 0 = s1 (parity of q0,q1), clbit 1 = s2 (parity of q1,q2).


def syndrome_extraction_circuit() -> QuantumCircuit:
    """Measure Z0Z1 into ancilla 3 and Z1Z2 into ancilla 4. Data is never measured."""
    qc = QuantumCircuit(5, 2, name="rep3_syndrome")
    qc.cx(0, 3)  # ancilla 3 flips if q0 = 1
    qc.cx(1, 3)  # ... and again if q1 = 1  -> ancilla 3 = q0 XOR q1
    qc.cx(1, 4)
    qc.cx(2, 4)  # ancilla 4 = q1 XOR q2
    qc.measure(3, 0)  # ancilla 3 -> clbit 0 (s1)
    qc.measure(4, 1)  # ancilla 4 -> clbit 1 (s2)
    return qc


def circuit_with_explicit_errors(logical_bit: int, error_qubits=()) -> QuantumCircuit:
    """Encode, apply chosen X errors (a test helper, not noise), extract syndrome."""
    qc = QuantumCircuit(5, 2)
    qc.compose(build_logical_basis_circuit(logical_bit), qubits=[0, 1, 2], inplace=True)
    for q in error_qubits:
        qc.x(q)
    qc.barrier()
    qc.compose(syndrome_extraction_circuit(), inplace=True)
    return qc