from dataclasses import replace

import numpy as np

from caqec.decoders.exact import ExactDecoder
from caqec.evaluation.decoder_metrics import evaluate, paired_difference
from caqec.simulation.dataset import DatasetConfig, generate_split


def run_scenario(scenario: dict, n: int, seed_seq, decoders: dict, base_cfg: DatasetConfig) -> dict:
    """Fresh data for one (p, q); decoders are frozen. `decoders` must contain 'lookup'.
    Adds 'oracle_aware', an exact decoder that knows this scenario's true noise (the ceiling)."""
    cfg = replace(base_cfg, p_values=(scenario["p"],), q=scenario["q"])
    data = generate_split(cfg, "train", n, seed_seq)  # "train" only names a trial-id offset; ids unused here
    X, y = data["X"], data["y"]
    aware = ExactDecoder(cfg)
    zeros = np.zeros(len(aware.joint), dtype=int)
    preds = {name: d.predict_flip(X) for name, d in decoders.items()}
    preds["oracle_aware"] = aware.predict_flip(X)
    return {"scenario": scenario, "n": int(len(y)), "y1_fraction": float(y.mean()),
            "analytic": {"lookup": aware.expected_failure_rate(zeros),
                         "oracle_aware": aware.expected_failure_rate()},
            "metrics": {name: evaluate(y, p) for name, p in preds.items()},
            "gain_vs_lookup": {name: paired_difference(y, preds["lookup"], p)
                               for name, p in preds.items() if name != "lookup"}}