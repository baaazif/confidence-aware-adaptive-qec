from qiskit import QuantumCircuit


def encoding_circuit() -> QuantumCircuit:
    """Spread qubit 0 across three qubits: a|0> + b|1> -> a|000> + b|111>."""
    qc = QuantumCircuit(3, name="rep3_encode")
    qc.cx(0, 1)  # control q0, target q1
    qc.cx(0, 2)  # control q0, target q2
    return qc


def build_logical_basis_circuit(logical_bit: int) -> QuantumCircuit:
    """Prepare logical 0 (|000>) or logical 1 (|111>)."""
    if logical_bit not in (0, 1):
        raise ValueError("logical_bit must be 0 or 1")
    qc = QuantumCircuit(3, name=f"rep3_logical_{logical_bit}")
    if logical_bit == 1:
        qc.x(0)  # state preparation: set q0 to |1>
    qc.compose(encoding_circuit(), inplace=True)
    return qc