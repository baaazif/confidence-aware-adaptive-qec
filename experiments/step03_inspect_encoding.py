import sys
from pathlib import Path

# Temporary: lets this script find src/caqec. We will replace this with a
# proper editable install later.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit.quantum_info import Statevector
from caqec.codes.repetition import build_logical_basis_circuit


def show(label, circuit):
    probs = Statevector(circuit).probabilities_dict()
    print(f"{label:28s} -> {probs}")


for bit in (0, 1):
    qc = build_logical_basis_circuit(bit)
    print(f"\nCircuit for logical {bit}:")
    print(qc.draw("text"))
    show(f"logical {bit}, no error", qc)

    for q in range(3):
        flipped = qc.copy()
        flipped.x(q)  # a manual single bit flip, not a noise model
        show(f"logical {bit}, X on qubit {q}", flipped)