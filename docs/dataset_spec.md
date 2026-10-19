# Dataset spec (exp02): syndrome history -> logical-flip label

Noise: 3 rounds; each data qubit flips w.p. p per round (accumulating); each syndrome
readout is wrong w.p. q; then a final perfect syndrome from the data readout.
p in {0.02, 0.05, 0.10}, q = 0.05. Per p: 20000 train, 5000 val, 5000 test trials.

Features: s1_r1, s2_r1, s1_r2, s2_r2, s1_r3, s2_r3, s1_final, s2_final (binary).
Label: y = 1 iff (true net data error) XOR (baseline correction for s_final) = 111.
Final correction = baseline correction XOR (y_hat * 111). y = 1 means baseline fails.

Rules: (1) label is ground truth; (2) ambiguous inputs are never relabeled, dropped
or majority-voted; (3) the label depends only on the net final error, so degenerate
histories share a label; (4) measurement errors change features, never labels;
(5) columns prefixed meta_ are never features; (6) fixed feature order.

Splits: three independent seed streams; unique trial ids; rows from one trial must
never be split. Identical input patterns across splits are expected (256 possible),
shared trials would be leakage. Any fitted preprocessing is fit on train only.
Limits: phenomenological noise only, no Z errors, no gate errors, d = 3, one seed.