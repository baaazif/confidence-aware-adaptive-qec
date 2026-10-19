import numpy as np

def logical_error_rate(p, d, shots, rng):
    flips = rng.random((shots, d)) < p
    decoded_wrong = flips.sum(axis=1) > d / 2
    return decoded_wrong.mean()

rng = np.random.default_rng(seed=1234)
d, shots = 3, 200_000
for p in [0.01, 0.05, 0.1, 0.3]:
    est = logical_error_rate(p, d, shots, rng)
    theory = 3 * p**2 - 2 * p**3
    print(f"p={p:.2f}  simulated p_L={est:.5f}  formula={theory:.5f}")