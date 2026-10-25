from math import ceil
from pathlib import Path

import numpy as np


def plot_failure_timeseries(panels: list, bin_size: int, path, cols: int = 4) -> None:
    """panels[i] = {title, curves: {decoder: (rate, lo, hi)}, change_start, change_end, median_alarm: {detector: t}}.
    Change markers come from the SIMULATOR's ground truth and are for plotting only."""
    import matplotlib.pyplot as plt
    rows = ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        for i, (name, (rate, lo, hi)) in enumerate(panel["curves"].items()):
            x = (np.arange(len(rate)) + 0.5) * bin_size
            ax.plot(x, rate, lw=1.2, color=f"C{i}", label=name)
            ax.fill_between(x, lo, hi, color=f"C{i}", alpha=0.15)
        cs, ce = panel.get("change_start"), panel.get("change_end")
        if cs is not None:
            ax.axvline(cs, ls="--", color="k", lw=1)
            if ce is not None and ce != cs:
                ax.axvspan(cs, ce, color="grey", alpha=0.12)
        for j, (dn, t) in enumerate(panel.get("median_alarm", {}).items()):
            if t is not None:
                ax.axvline(t, ls=":", color=f"C{len(panel['curves']) + j}", lw=1.3, label=f"{dn}: median alarm")
        ax.set_title(panel["title"], fontsize=9)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6)
    fig.supxlabel("shot index")
    fig.supylabel("logical failure rate per time bin (pooled over trials)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detection_summary(rows: list, path) -> None:
    """rows[i] = {label, cusum: summary, window: summary}. Left: delay (median, whisker to p90). Right: P(detect)."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(rows))
    for j, det in enumerate(("cusum", "window")):
        off = (j - 0.5) * 0.36
        med = np.array([(r[det].get("delay_median") or 0.0) for r in rows])
        p90 = np.array([(r[det].get("delay_p90") or 0.0) for r in rows])
        a1.bar(x + off, med, 0.36, yerr=[np.zeros(len(rows)), np.maximum(p90 - med, 0)], capsize=3,
               color=f"C{j}", label=det)
        pd = np.array([r[det]["p_detect"] for r in rows])
        lo = np.array([r[det]["p_detect_ci"][0] for r in rows])
        hi = np.array([r[det]["p_detect_ci"][1] for r in rows])
        a2.bar(x + off, pd, 0.36, yerr=[np.maximum(pd - lo, 0), np.maximum(hi - pd, 0)], capsize=3, color=f"C{j}", label=det)
    for a, yl, t in ((a1, "detection delay in shots (median, whisker = 90th percentile)", "Detection delay"),
                     (a2, "fraction of streams in which the change was detected", "Detection probability")):
        a.set_xticks(x)
        a.set_xticklabels([r["label"] for r in rows], rotation=30, ha="right", fontsize=8)
        a.set_ylabel(yl, fontsize=8)
        a.set_title(t)
        a.grid(alpha=0.3, axis="y")
        a.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)