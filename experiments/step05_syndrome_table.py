import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit import transpile
from qiskit_aer import AerSimulator

from caqec.codes.syndrome import circuit_with_explicit_errors
from caqec.decoders.lookup import apply_correction, majority_vote, parse_syndrome

sim = AerSimulator()
print("Logical 0 (000). Error pattern shown as (b0, b1, b2); 1 = flipped.\n")
print(f"{'errors':8s} {'raw key':8s} {'(s1,s2)':8s} {'corrected':10s} {'majority':9s} result")
for pattern in product((0, 1), repeat=3):
    qubits = [q for q, b in enumerate(pattern) if b]
    qc = circuit_with_explicit_errors(0, qubits)
    counts = sim.run(transpile(qc, sim), shots=100).result().get_counts()
    (key,) = counts  # exactly one outcome expected: the syndrome is deterministic
    syn = parse_syndrome(key)
    corrected = apply_correction(pattern, syn)  # data after error = pattern (logical 0)
    ok = corrected == (0, 0, 0)
    print(f"{str(pattern):8s} {key:8s} {str(syn):8s} {str(corrected):10s} "
          f"{majority_vote(pattern):<9d} {'ok' if ok else 'LOGICAL ERROR'}")