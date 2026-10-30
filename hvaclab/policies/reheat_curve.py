"""Decouple the reheat valve characteristic from the V4 zone damper command."""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class ReheatCurvePolicy(CoordinatedHeatPolicy):
    def __init__(self, *args, reheat_exponent=1., **kwargs):
        if not np.isfinite(reheat_exponent) or reheat_exponent <= 0:
            raise ValueError("reheat exponent must be finite and positive")
        super().__init__(*args,**kwargs)
        self.reheat_exponent=float(reheat_exponent)

    def _command_zones(self, continuous, zones, obs, low, high):
        super()._command_zones(continuous,zones,obs,low,high)
        if self.reheat_exponent == 1. or not (self.zone_pi_enabled and self._pi_active):
            return
        for vav,_,indices in zones:
            valve=indices.get("reheat_valve_percentage_command")
            if valve is None:
                continue
            demand=max(0.,-self._zone_u.get(vav,0.))
            continuous[valve]=np.float32(2.*demand**self.reheat_exponent-1.)
