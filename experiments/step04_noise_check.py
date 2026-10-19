import sys
from math import comb, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit import transpile
from qiskit_aer import AerSimulator

from caqec.codes.repetition import build_logical_basis_circuit
from caqec.noise.bitflip import add_noise_slot, build_bitflip_noise_model

SHOTS = 100_000
SEED = 1234


def run_counts(p, shots=SHOTS, seed=SEED):
    qc = add_noise_slot(build_logical_basis_circuit(0))  # starts as 000
    qc.measure_all()
    sim = AerSimulator(noise_model=build_bitflip_noise_model(p))
    tqc = transpile(qc, sim, optimization_level=0)  # level 0: keep the id gates
    return sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()


if __name__ == "__main__":
    for p in (0.0, 0.1, 1.0):
        counts = run_counts(p)
        print(f"\np = {p}")
        print("  counts:", dict(sorted(counts.items())))
        for q in range(3):
            # little-endian: qubit q is character index (2 - q) in the string
            flips = sum(n for s, n in counts.items() if s[2 - q] == "1")
            se = sqrt(p * (1 - p) / SHOTS)
            print(f"  qubit {q}: flip freq {flips / SHOTS:.5f}  expected {p}  (std err {se:.5f})")
        print("  number of flips: observed vs Binomial(3, p)")
        for k in range(4):
            obs = sum(n for s, n in counts.items() if s.count("1") == k) / SHOTS
            theory = comb(3, k) * p**k * (1 - p) ** (3 - k)
            print(f"    k={k}: observed {obs:.5f}  theory {theory:.5f}")