from pathlib import Path

import numpy as np


def plot_series(series: dict, path, xlabel: str, ylabel: str, title: str, xlog=False, ylog=False,
                shade=None, zero_line=False) -> None:
    """series[name] = list of (x, value, ci_low, ci_high), sorted by x."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (name, pts) in enumerate(series.items()):
        x, y, lo, hi = (np.array(c, dtype=float) for c in zip(*pts))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3, ms=4, lw=1, color=f"C{i}", label=name)
    if shade:
        ax.axvspan(shade[0], shade[1], color="grey", alpha=0.12, label="trained range")
    if zero_line:
        ax.axhline(0, color="k", lw=1)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)