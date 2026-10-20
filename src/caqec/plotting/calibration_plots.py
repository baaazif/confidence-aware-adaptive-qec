from pathlib import Path

import numpy as np


def plot_reliability(tables: dict, path, title: str = "") -> None:
    """tables[name] = reliability_table rows. Diagonal = perfect calibration."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for i, (name, rows) in enumerate(tables.items()):
        r = [x for x in rows if x["n"] > 0]
        x = np.array([a["mean_pred"] for a in r])
        y = np.array([a["observed"] for a in r])
        lo = np.array([a["ci_low"] for a in r])
        hi = np.array([a["ci_high"] for a in r])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3, ms=4, lw=1, color=f"C{i}", label=name)
    ax.set_xlabel("mean predicted P(y = 1) in bin")
    ax.set_ylabel("observed frequency of y = 1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(curves: dict, path, title: str = "", reference: dict = None) -> None:
    """curves[name] = risk_coverage points. reference[name] = failure rate at full coverage (no confidence)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, (name, pts) in enumerate(curves.items()):
        x = np.array([p["coverage"] for p in pts])
        y = np.array([p["failure_rate"] for p in pts])
        lo = np.array([p["ci_low"] for p in pts])
        hi = np.array([p["ci_high"] for p in pts])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=2, ms=3, lw=1, color=f"C{i}", label=name)
    for j, (name, rate) in enumerate((reference or {}).items()):
        ax.plot([1.0], [rate], marker="s", ls="", color=f"C{len(curves) + j}", label=f"{name} (no confidence)")
    ax.set_xlabel("coverage: fraction of shots accepted")
    ax.set_ylabel("logical failure rate among accepted shots")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)