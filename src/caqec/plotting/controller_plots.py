from math import ceil
from pathlib import Path

import numpy as np


def plot_method_costs(panels: list, path, cols: int = 4) -> None:
    """panels[i] = {title, methods: {name: (mean, ci_low, ci_high)}}: total cost per stream, own y-axis per panel."""
    import matplotlib.pyplot as plt
    rows = ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        names = list(panel["methods"])
        for i, n in enumerate(names):
            m, lo, hi = panel["methods"][n]
            ax.errorbar(i, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], fmt="o", capsize=3, color=f"C{i}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(panel["title"], fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.supylabel("total cost per 6000-shot stream (failure-equivalents; mean, 95% CI)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(rows: dict, scenarios: list, path) -> None:
    """rows[method][scenario] = (diff, ci_low, ci_high) of cost(method) - cost(proposed). Above 0 = proposed cheaper."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    names = list(rows)
    width = 0.8 / max(len(names), 1)
    for j, n in enumerate(names):
        xs = np.arange(len(scenarios)) + (j - (len(names) - 1) / 2) * width
        d = np.array([rows[n][s][0] for s in scenarios])
        lo = np.array([rows[n][s][1] for s in scenarios])
        hi = np.array([rows[n][s][2] for s in scenarios])
        ax.errorbar(xs, d, yerr=[np.maximum(d - lo, 0), np.maximum(hi - d, 0)], fmt="o", capsize=2, ms=4,
                    color=f"C{j}", label=n)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cost(method) - cost(proposed), mean and 95% CI\n(above 0: the proposed controller is cheaper)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)