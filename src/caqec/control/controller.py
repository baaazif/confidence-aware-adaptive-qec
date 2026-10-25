"""Rule-based adaptive controllers. They receive three numbers per shot, in time order, all derived from the
observable syndromes: detection-event count, last-round mismatch count, and the monitor decoder's self-estimated
failure probability. Never labels, never the change time, never the true noise level.

Actions: continue (do nothing), wait (collect more evidence), switch (to a prepared decoder, immediate), update
(rebuild an exact decoder for the estimated noise; takes effect after a delay and costs more)."""
from dataclasses import dataclass, field

from caqec.control.costs import CostModel
from caqec.control.knowledge import NoiseKnowledge
from caqec.detection.online import WindowDetector

STATIC = ("lookup", "minweight", "tree")


@dataclass(frozen=True)
class ControllerConfig:
    name: str = "proposed"
    evidence_shots: int = 300       # shots collected after an alarm before deciding (minimum evidence)
    horizon: int = 3000             # shots over which a benefit must pay back the adaptation cost
    margin: float = 0.0             # required net benefit (failure-equivalents)
    dwell: int = 1000               # minimum shots between adaptations
    use_guard: bool = True          # veto when the two noise estimates disagree (model-consistency check)
    max_waits: int = 1              # extra evidence rounds allowed when the guard disagrees
    estimate_q: bool = True         # estimate q from the last-round mismatch instead of assuming it
    cost_aware: bool = True         # include adaptation and compute costs in the decision
    robust_ci: bool = True          # require the benefit to hold across the confidence interval of p
    allow_update: bool = True
    initial_decision: bool = True   # decide once at the end of the warm-up
    z: float = 2.0
    detector: tuple = (500, 100, 3.5)   # (reference shots, window shots, z threshold); calibrated in stage 8
    costs: CostModel = field(default_factory=CostModel)


class FixedController:
    def __init__(self, spec: str):
        self.spec, self.n_switch, self.n_update, self.log = spec, 0, 0, []

    def spec_for(self, t):
        return self.spec

    def observe(self, t, e, mm, sp):
        pass


class AdaptiveController:
    def __init__(self, knowledge: NoiseKnowledge, cfg: ControllerConfig = ControllerConfig(), q_nominal: float = 0.05,
                 initial: str = "tree"):
        self.k, self.cfg, self.q_nom = knowledge, cfg, q_nominal
        self.det = WindowDetector(*cfg.detector)
        self.current, self.pending = initial, None
        self.n_switch = self.n_update = 0
        self.log = []
        self.mode, self.last_action = "idle", -10**9
        self._initial_done = not cfg.initial_decision
        self._clear()

    def _clear(self):
        self.n = self.waits = 0
        self.s_e = self.s_e2 = self.s_m = self.s_s = self.s_s2 = 0.0

    def _add(self, e, mm, sp):
        self.n += 1
        self.s_e += e; self.s_e2 += e * e; self.s_m += mm; self.s_s += sp; self.s_s2 += sp * sp

    def spec_for(self, t):
        if self.pending and t >= self.pending[0]:
            self.current, self.pending = self.pending[1], None
        return self.current

    def observe(self, t, e, mm, sp):
        alarm = self.det.update(e)
        if self.mode == "idle":
            if alarm:
                self.log.append({"t": t, "event": "alarm"})
                self.mode, self.target = "evidence", self.cfg.evidence_shots
                self._clear()
                return
            if not self._initial_done:
                self._add(e, mm, sp)
                if t + 1 >= self.cfg.detector[0]:
                    self._initial_done = True
                    self._decide(t, "initial")
            return
        self._add(e, mm, sp)
        if self.n >= self.target:
            self._decide(t, "alarm")

    # ---- decision rule ----
    def _estimate(self):
        n = self.n
        mean = lambda s: s / n
        sd = lambda s, s2: max(s2 / n - (s / n) ** 2, 1e-12) ** 0.5
        return self.k.estimate(n, mean(self.s_e), sd(self.s_e, self.s_e2), mean(self.s_m), mean(self.s_s),
                               sd(self.s_s, self.s_s2), self.cfg.z, None if self.cfg.estimate_q else self.q_nom)

    def _decide(self, t, why):
        cfg, est = self.cfg, self._estimate()
        info = {"t": t, "event": "decision", "why": why, "n": self.n, **{k: round(v, 4) for k, v in est.items()}}
        if cfg.use_guard and (est["p_hi_ev"] < est["p_lo_cf"] or est["p_hi_cf"] < est["p_lo_ev"]):
            if self.waits < cfg.max_waits:
                self.waits += 1
                self.target = self.n + cfg.evidence_shots                 # action: wait for more evidence
                info["action"] = "wait"
                self.log.append(info)
                return
            info["action"] = "continue (guard veto)"
            return self._finish(info)
        if t - self.last_action < cfg.dwell:
            info["action"] = "continue (dwell)"
            return self._finish(info)
        best = self._best_option(est)
        if best is None:
            info["action"] = "continue"
            return self._finish(info)
        benefit, name, spec = best
        info.update(action=name, spec=str(spec), benefit=round(benefit, 3))
        self.last_action = t
        if name == "update":
            self.pending, self.n_update = (t + cfg.costs.update_delay, spec), self.n_update + 1
        else:
            self.current, self.n_switch = spec, self.n_switch + 1
        self._finish(info)

    def _finish(self, info):
        self.log.append(info)
        self.mode = "idle"
        self._clear()

    def _best_option(self, est):
        cfg, costs = self.cfg, self.cfg.costs
        q = est["q_hat"]
        ps = [est["p_lo_ev"], est["p_hat_ev"], est["p_hi_ev"]] if cfg.robust_ci else [est["p_hat_ev"]]
        rows = []
        for name in STATIC + (("update",) if cfg.allow_update else ()):
            spec = name if name != "update" else ("bayes", *self.k.quant(est["p_hat_ev"], q))
            if spec == self.current:
                continue
            horizon = cfg.horizon if name != "update" else max(cfg.horizon - costs.update_delay, 0)
            adapt = (costs.update_cost if name == "update" else costs.switch_cost) if cfg.cost_aware else 0.0
            extra = (costs.decode(spec) - costs.decode(self.current)) if cfg.cost_aware else 0.0
            gains = [costs.c_fail * horizon * (self.k.failure(self.current, p, q) - self.k.failure(spec, p, q))
                     - horizon * extra - adapt for p in ps]
            rows.append((min(gains), name, spec))
        if not rows:
            return None
        best = max(rows, key=lambda r: r[0])
        return best if best[0] > cfg.margin else None


# Adaptive BASELINE: change detection + immediate switching. No evidence waiting beyond 100 shots, no guard,
# no q estimation, no dwell, no cost logic, no update action, point estimate only.
BASELINE_CONFIG = ControllerConfig(name="adaptive_baseline", evidence_shots=100, dwell=0, use_guard=False,
                                   estimate_q=False, cost_aware=False, robust_ci=False, allow_update=False)