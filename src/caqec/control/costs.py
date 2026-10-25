"""Adaptation-cost model. ALL numbers are ASSUMED, in 'logical-failure equivalents', and configurable.
They are not measurements of any real system."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostModel:
    c_fail: float = 1.0                 # one logical failure
    switch_cost: float = 5.0            # switching between prepared decoders (assumed)
    update_cost: float = 40.0           # rebuilding a decoder for newly estimated noise (assumed)
    update_delay: int = 200             # shots until an update takes effect; the old decoder keeps running
    monitor_cost_per_shot: float = 0.001  # noise estimation + confidence monitoring (assumed)
    decode_cost_per_shot: dict = field(default_factory=lambda: {
        "lookup": 0.0, "minweight": 0.0005, "tree": 0.002, "updated": 0.0005})  # assumed relative compute

    def decode(self, spec) -> float:
        return self.decode_cost_per_shot["updated" if isinstance(spec, tuple) else spec]