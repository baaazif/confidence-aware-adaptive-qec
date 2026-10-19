# Experiment 01: logical failure with vs without correction

Code: 3-qubit repetition code, logical 0 = 000.
Noise: independent bit flips with probability p per qubit (code-capacity;
encoding, syndrome measurement and correction are perfect).
Arms: (a) one unprotected qubit; (b) three qubits with syndrome decoding.
Trial: prepare logical 0, apply noise, decode (arm b), record failure.
Metric: failure rate = failures / N, with 95% CI.
Reference: unprotected = p; protected = 3p^2 - 2p^3.
Settings: p in {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, N = 200000, fixed seed.
Expected: protected < unprotected for p < 0.5; equal at p = 0.5.
Not shown: phase errors, measurement noise, circuit-level noise,
thresholds, anything about AI or confidence.