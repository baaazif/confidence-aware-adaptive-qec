from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2, 2)
qc.h(0)               # put qubit 0 into an equal superposition
qc.cx(0, 1)           # entangle qubit 1 with qubit 0
qc.measure([0, 1], [0, 1])

print(qc.draw("text"))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=1000, seed_simulator=1234).result()
print(result.get_counts())