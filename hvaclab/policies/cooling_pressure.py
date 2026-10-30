"""Bounded pressure response to hot zones with saturated cooling PI demand.

Only a running cooling handler can receive a higher pressure request. The
response grows smoothly with measured upper-band excursion. A mixed-air
estimate vetoes extra airflow when intake mixing would be below the comfort
lower bound. No temperature sensor is treated as actual AHU supply air.
"""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class CoolingPressurePolicy(CoordinatedHeatPolicy):
    def __init__(self, *args, cooling_pressure_ceiling=None,
                 cooling_pressure_floor=-0.8, cooling_effort_threshold=0.8,
                 cooling_error_full_k=1.0, **kwargs):
        if not np.isfinite(cooling_pressure_floor) or not -1 <= cooling_pressure_floor <= 1:
            raise ValueError("invalid normalized pressure floor")
        if cooling_pressure_ceiling is not None and (
                not np.isfinite(cooling_pressure_ceiling)
                or not cooling_pressure_floor <= cooling_pressure_ceiling <= 1):
            raise ValueError("invalid normalized pressure ceiling")
        if not 0 <= cooling_effort_threshold <= 1:
            raise ValueError("invalid PI effort threshold")
        if not np.isfinite(cooling_error_full_k) or cooling_error_full_k <= 0:
            raise ValueError("invalid full-response error")
        super().__init__(*args, **kwargs)
        self.cooling_pressure_ceiling = cooling_pressure_ceiling
        self.cooling_pressure_floor = float(cooling_pressure_floor)
        self.cooling_effort_threshold = float(cooling_effort_threshold)
        self.cooling_error_full_k = float(cooling_error_full_k)
        self.pressure_boosts = 0

    def get_action(self, obs):
        action = super().get_action(obs)
        if self.cooling_pressure_ceiling is None or not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        low, high = self._comfort_band
        for ctrl in self.ahu_controllers:
            pressure, oat, recirc = (ctrl.get(k) for k in ("pressure_idx", "oat_idx", "recirc_idx"))
            if (pressure is None or oat is None or recirc is None or not ctrl["zones"]
                    or not disc[ctrl["run_discrete_idx"]] or cont[ctrl["temp_idx"]] >= 0):
                continue
            temperatures = [float(self._denormalize_zone_temp(obs[index]))
                            for _, index, _ in ctrl["zones"]]
            if not np.all(np.isfinite(temperatures)) or not np.isfinite(obs[oat]):
                continue
            ratio = float(np.clip(.5*(float(cont[recirc])+1.), 0., self.max_recirculation))
            mixed = ratio*float(np.mean(temperatures)) + (1.-ratio)*float(obs[oat])
            if mixed < low:
                continue
            hot_error = max((temperature-high for temperature, (vav, _, _) in
                             zip(temperatures, ctrl["zones"])
                             if self._zone_u.get(vav, 0.) >= self.cooling_effort_threshold), default=0.)
            if hot_error <= 0:
                continue
            fraction = min(1., hot_error/self.cooling_error_full_k)
            requested = self.cooling_pressure_floor + fraction*(
                self.cooling_pressure_ceiling-self.cooling_pressure_floor)
            if requested > float(cont[pressure]):
                cont[pressure] = np.float32(requested)
                self.pressure_boosts += 1
        return action
