"""Comparison plots. Call matplotlib.use("Agg") in the script if no window is wanted."""
from pathlib import Path

import numpy as np


def _label(g):
    return "all rows" if g == "all" else f"p = {g}"


def plot_failure_rates(by_decoder: dict, groups: list, path, title: str = "") -> None:
    """One panel per group; one point per decoder with its 95% interval. Panels have their own y-axis."""
    import matplotlib.pyplot as plt
    names = list(by_decoder)
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 4.4))
    for ax, g in zip(np.atleast_1d(axes), groups):
        for i, name in enumerate(names):
            m = by_decoder[name][g]
            ax.errorbar(i, m["failure_rate"], yerr=[[m["failure_rate"] - m["ci_low"]],
                                                    [m["ci_high"] - m["failure_rate"]]],
                        fmt="o", capsize=4, color=f"C{i}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(_label(g))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("logical failure rate")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paired_differences(paired: dict, models: list, groups: list, path, title: str = "") -> None:
    """paired[group][model] = {diff, ci_low, ci_high}: failure(lookup) - failure(model)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.2
    for j, model in enumerate(models):
        xs = np.arange(len(groups)) + (j - (len(models) - 1) / 2) * width
        d = [paired[g][model] for g in groups]
        y = np.array([m["diff"] for m in d])
        ax.errorbar(xs, y, yerr=[y - [m["ci_low"] for m in d], [m["ci_high"] for m in d] - y],
                    fmt="o", capsize=4, color=f"C{j + 1}", label=model)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_label(g) for g in groups])
    ax.set_ylabel("failure(lookup) - failure(model)\n(positive = model better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)