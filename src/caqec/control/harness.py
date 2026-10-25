"""Runs a controller over a stream and accounts for failures and costs.
Ground truth (y) is used ONLY here, after the fact, to count failures. Controllers never see it."""
from time import perf_counter

import numpy as np

from caqec.control.costs import CostModel
from caqec.decoders.base import pattern_codes
from caqec.detection.online import defect_counts


def observables(X, selfp_table):
    """Per-shot observables from syndrome features X (..., 2R+2): events, last-round mismatch, monitor self-failure."""
    X = np.asarray(X)
    R = (X.shape[-1] - 2) // 2
    mismatch = (X[..., -2:] ^ X[..., 2 * R - 2: 2 * R]).sum(axis=-1).astype(float)
    codes = pattern_codes(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1])
    return defect_counts(X), mismatch, selfp_table[codes], codes


def run_controller(controller, events, mismatch, selfp, codes, y, knowledge, costs: CostModel, monitored: bool) -> dict:
    T = len(events)
    ev, mm, sp = events.tolist(), mismatch.tolist(), selfp.tolist()
    ids, specs, index = np.empty(T, dtype=np.int32), [], {}
    t0 = perf_counter()
    for t in range(T):
        spec = controller.spec_for(t)
        i = index.get(spec)
        if i is None:
            i = index[spec] = len(specs)
            specs.append(spec)
        ids[t] = i
        controller.observe(t, ev[t], mm[t], sp[t])
    runtime = perf_counter() - t0
    wrong = np.zeros(T, dtype=bool)
    decode_cost = 0.0
    for i, spec in enumerate(specs):
        m = ids == i
        wrong[m] = knowledge.table(spec)[codes[m]] != y[m]
        decode_cost += costs.decode(spec) * int(m.sum())
    failures = int(wrong.sum())
    parts = {"failures": costs.c_fail * failures,
             "switching": costs.switch_cost * controller.n_switch,
             "updates": costs.update_cost * controller.n_update,
             "compute": decode_cost + (costs.monitor_cost_per_shot * T if monitored else 0.0)}
    return {"wrong": wrong, "cost": sum(parts.values()), "parts": parts, "n_switch": controller.n_switch,
            "n_update": controller.n_update, "log": controller.log, "runtime_s": runtime}