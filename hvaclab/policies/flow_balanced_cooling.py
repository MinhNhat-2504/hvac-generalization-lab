"""Cooling-pressure boost with non-target damper flow compensation.

For the shipped Q proportional to damper*sqrt(P) law, reduce non-target
openings by sqrt(P_old/P_new). The ventilation floor can prevent exact
compensation; an optional guard vetoes boosts near cold minimum-flow zones.
"""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class FlowBalancedCoolingPolicy(CoordinatedHeatPolicy):
    def __init__(self, *args, cooling_pressure_ceiling=None,
                 cooling_pressure_floor=-0.8, cooling_effort_threshold=0.8,
                 cooling_error_full_k=1.0, protect_minimum_flow=False,
                 minimum_damper=0.1, cold_neighbor_margin_k=0.5, **kwargs):
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
        if not 0 < minimum_damper <= 1:
            raise ValueError("minimum damper must be in (0,1]")
        if not np.isfinite(cold_neighbor_margin_k) or cold_neighbor_margin_k < 0:
            raise ValueError("cold-neighbor margin must be nonnegative")
        super().__init__(*args, **kwargs)
        self.cooling_pressure_ceiling = cooling_pressure_ceiling
        self.cooling_pressure_floor = float(cooling_pressure_floor)
        self.cooling_effort_threshold = float(cooling_effort_threshold)
        self.cooling_error_full_k = float(cooling_error_full_k)
        self.pressure_boosts = 0
        self.protect_minimum_flow = bool(protect_minimum_flow)
        self.minimum_damper = float(minimum_damper)
        self.cold_neighbor_margin_k = float(cold_neighbor_margin_k)
        self.flow_compensations = 0
        self.flow_guard_vetoes = 0

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
                # Native pressure is zero-based and proportional to (action+1)
                # in the shipped normalization. Clamp to the running floor.
                old_pressure = max(.5*(float(cont[pressure])+1.),
                                   .5*(self.cooling_pressure_floor+1.))
                new_pressure = max(.5*(requested+1.),
                                   .5*(self.cooling_pressure_floor+1.))
                if old_pressure <= 0 or new_pressure <= 0:
                    continue
                factor = float(np.sqrt(old_pressure/new_pressure))
                updates = []
                veto = False
                for temperature, (vav, _, indices) in zip(temperatures, ctrl["zones"]):
                    if temperature > high and self._zone_u.get(vav,0.) >= self.cooling_effort_threshold:
                        continue  # The hot target keeps its requested opening.
                    damper = indices.get("supply_air_damper_percentage_command")
                    if damper is None:
                        veto = True
                        break
                    opening = float(np.clip(.5*(float(cont[damper])+1.), self.minimum_damper, 1.))
                    scaled = opening*factor
                    if (self.protect_minimum_flow and scaled < self.minimum_damper
                            and temperature <= low+self.cold_neighbor_margin_k):
                        veto = True
                        break
                    compensated = max(self.minimum_damper,scaled)
                    if compensated < opening-1e-12:
                        updates.append((damper,2.*compensated-1.))
                if veto:
                    self.flow_guard_vetoes += 1
                    continue
                cont[pressure] = np.float32(requested)
                for index,value in updates:
                    cont[index] = np.float32(value)
                self.flow_compensations += len(updates)
                self.pressure_boosts += 1
        return action
