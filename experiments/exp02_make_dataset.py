import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from caqec.evaluation.dataset_checks import run_all_checks
from caqec.evaluation.stats import wilson_interval
from caqec.simulation.dataset import (DatasetConfig, generate_all, save_split, write_card)

cfg = DatasetConfig()
out_dir = ROOT / "data" / "exp02"
out_dir.mkdir(parents=True, exist_ok=True)

splits = generate_all(cfg)
for name, arrays in splits.items():
    save_split(out_dir / f"{name}.npz", arrays)
write_card(out_dir / "dataset_card.json", cfg, splits)
print(f"numpy {np.__version__}; config: {cfg}\nsaved to {out_dir.relative_to(ROOT)}\n")

print("Baseline lookup decoder failure rate on each split (= fraction of rows with y = 1):")
for name in ("train", "val", "test"):
    s = splits[name]
    for p in cfg.p_values:
        m = np.isclose(s["meta_p"], p)
        k, n = int(s["y"][m].sum()), int(m.sum())
        lo, hi = wilson_interval(k, n)
        print(f"  {name:5s} p={p:<5} rows={n:6d}  y=1: {k / n:.4f} [{lo:.4f}, {hi:.4f}]")

print("\nValidation checks:")
checks = run_all_checks(splits, cfg)
for c in checks:
    print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}\n         {c.detail}")
failed = [c for c in checks if not c.ok]
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
sys.exit(1 if failed else 0)