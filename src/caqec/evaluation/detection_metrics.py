"""Detection scoring. Uses the TRUE change time, so it is analysis code, not part of any online method."""
import numpy as np

from caqec.evaluation.stats import wilson_interval


def score_detection(alarms, T: int, warmup: int, change_start=None, max_delay=None) -> dict:
    """false alarm = alarm before the change (or any alarm if there is no change);
    detected = an alarm at/after the change within max_delay shots; delay = first such alarm - change."""
    alarms = sorted(alarms)
    if change_start is None:
        return {"false_alarms": len(alarms), "exposure": max(T - warmup, 0), "detected": None, "delay": None}
    first = next((a for a in alarms if a >= change_start), None)
    delay = None if first is None else first - change_start
    detected = first is not None and (max_delay is None or delay <= max_delay)
    return {"false_alarms": sum(a < change_start for a in alarms), "exposure": max(change_start - warmup, 0),
            "detected": bool(detected), "delay": delay if detected else None}


def summarize_detection(scores: list) -> dict:
    """Aggregate per-stream scores of one scenario and one detector."""
    n = len(scores)
    fa, exposure = sum(s["false_alarms"] for s in scores), sum(s["exposure"] for s in scores)
    lo, hi = wilson_interval(sum(s["false_alarms"] > 0 for s in scores), n)
    out = {"streams": n, "false_alarms": fa, "false_alarms_per_1000_shots": 1000 * fa / exposure if exposure else None,
           "p_any_false_alarm": sum(s["false_alarms"] > 0 for s in scores) / n, "p_any_fa_ci": [lo, hi]}
    if scores[0]["detected"] is not None:
        det = [s for s in scores if s["detected"]]
        dlo, dhi = wilson_interval(len(det), n)
        delays = np.array([s["delay"] for s in det], dtype=float)
        out.update({"p_detect": len(det) / n, "p_detect_ci": [dlo, dhi], "missed": n - len(det),
                    "delay_mean": float(delays.mean()) if len(delays) else None,
                    "delay_median": float(np.median(delays)) if len(delays) else None,
                    "delay_p90": float(np.percentile(delays, 90)) if len(delays) else None})
    return out