"""Experimental post-coordination pump cap, with an optional comfort guard.

Lower head also lowers water flow, so this is not a guaranteed energy saving
at unchanged comfort. The guarded arm restores the reference request when
an open reheat valve serves a zone near the lower comfort edge or approaches
full opening. Inputs are current observations and the final e_sync commands.
"""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class GuardedPumpPolicy(CoordinatedHeatPolicy):
    def __init__(self, *args, pump_limit=None, guard_pump=True,
                 pump_reserve_k=0.5, pump_valve_limit=0.8, **kwargs):
        if pump_limit is not None and (not np.isfinite(pump_limit) or not -1 <= pump_limit <= 1):
            raise ValueError("pump_limit must be a finite normalized action")
        if not np.isfinite(pump_reserve_k) or pump_reserve_k < 0:
            raise ValueError("pump reserve must be finite and nonnegative")
        if not 0 < pump_valve_limit <= 1:
            raise ValueError("valve limit must be in (0, 1]")
        super().__init__(*args, **kwargs)
        self.pump_limit = pump_limit
        self.guard_pump = bool(guard_pump)
        self.pump_reserve_k = float(pump_reserve_k)
        self.pump_valve_limit = float(pump_valve_limit)
        self.pump_caps = 0
        self.pump_guard_restores = 0

    def get_action(self, obs):
        action = super().get_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        if (self.pump_limit is None or self.water_pressure_idx is None
                or not 0 <= self.water_discrete_idx < len(disc)
                or not disc[self.water_discrete_idx]):
            return action
        opened = []
        for ctrl in self.ahu_controllers:
            for _, index, indices in ctrl["zones"]:
                valve = indices.get("reheat_valve_percentage_command")
                if valve is None:
                    continue
                opening = .5 * (float(cont[valve]) + 1.)
                if opening > 0:
                    opened.append((self._denormalize_zone_temp(obs[index]), opening))
        # Closed valves already imply zero flow in the pump model. Do not
        # describe lowering a command there as a physical energy saving.
        if not opened:
            return action
        if self.guard_pump and any(
                not np.isfinite(temp) or temp <= self._comfort_band[0] + self.pump_reserve_k
                or opening >= self.pump_valve_limit for temp, opening in opened):
            self.pump_guard_restores += 1
            return action
        if float(cont[self.water_pressure_idx]) > self.pump_limit:
            cont[self.water_pressure_idx] = np.float32(self.pump_limit)
            self.pump_caps += 1
        return action
