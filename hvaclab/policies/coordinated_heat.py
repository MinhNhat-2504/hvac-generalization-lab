"""Coordinate reheat airflow and water, optionally bounding ventilation losses.

The original d_heat experiment improves mild heating scenarios but hurts deep
cold, and frequently starts a fan with its water pump stopped. These are two
independent levers, exposed separately for a factorial ablation. Existing d_*
variants retain their original behavior.
"""

import numpy as np

from hvaclab.policies.heat_delivery import HeatDeliveryPolicy


class CoordinatedHeatPolicy(HeatDeliveryPolicy):
    """Supply water on airflow overrides and reject inefficient extra ventilation.

    The optional guard estimates the *best possible* fraction of coil heat
    reaching a zone: (discharge_limit - target)/(discharge_limit - mixed_air).
    With very cold intake air much of the coil duty merely warms ventilation
    air. A low fraction vetoes this extra run; the inherited heating controller
    still operates normally. This is a hypothesis to measure, not a capacity
    guarantee. The 35 C discharge and 70% recirculation limits are the shipped
    simulator defaults, exposed explicitly rather than inferred from labels.
    """

    def __init__(self, *args, coordinate_water=True,
                 minimum_delivery_efficiency=None, discharge_limit_k=308.15,
                 max_recirculation=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        if minimum_delivery_efficiency is not None and not (
            0.0 <= minimum_delivery_efficiency <= 1.0
        ):
            raise ValueError("minimum_delivery_efficiency must be in [0, 1]")
        if not 0.0 <= max_recirculation <= 1.0:
            raise ValueError("max_recirculation must be in [0, 1]")
        if not np.isfinite(discharge_limit_k):
            raise ValueError("discharge_limit_k must be finite")
        self.coordinate_water = bool(coordinate_water)
        self.minimum_delivery_efficiency = minimum_delivery_efficiency
        self.discharge_limit_k = float(discharge_limit_k)
        self.max_recirculation = float(max_recirculation)
        self.heat_guard_vetoes = 0
        self.water_sync_steps = 0

    @property
    def name(self):
        return "coordinated_heat"

    def _allow_heat_override(self, obs, ctrl, demands):
        if self.coordinate_water and (
            self.water_temp_idx is None or self.water_pressure_idx is None
            or not 0 <= self.water_discrete_idx < self.n_discrete
        ):
            return False
        if self.minimum_delivery_efficiency is None:
            return True
        idx = ctrl.get("oat_idx")
        # Outside-air observations are raw kelvin, as in the inherited
        # economizer. Missing/non-finite weather cannot justify an extra run.
        if idx is None or not np.isfinite(obs[idx]):
            self.heat_guard_vetoes += 1
            return False
        return_temp = float(np.mean([
            self._denormalize_zone_temp(obs[i]) for i in ctrl["zone_obs_indices"]
        ]))
        target = 0.5 * sum(self._comfort_band)
        mixed = (self.max_recirculation * return_temp
                 + (1.0 - self.max_recirculation) * float(obs[idx]))
        lift = self.discharge_limit_k - target
        coil_lift = self.discharge_limit_k - mixed
        if not np.isfinite(return_temp) or lift <= 0.0:
            allowed = False
        else:
            efficiency = lift / max(coil_lift, lift)
            allowed = efficiency >= self.minimum_delivery_efficiency
        if not allowed:
            self.heat_guard_vetoes += 1
        return allowed

    def get_action(self, obs):
        before = self.heat_starts
        action = super().get_action(obs)
        if not self.coordinate_water or self.heat_starts == before:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        disc[self.water_discrete_idx] = 1
        # Reuse the inherited demand reset, updated once by the parent this
        # step. Never cool an already-hot loop or bypass the pump economy cap.
        shortfall = self._heating_shortfall(obs, *self._comfort_band)
        requested = 2.0 * self._water_fraction(shortfall) - 1.0
        cont[self.water_temp_idx] = np.float32(max(
            float(cont[self.water_temp_idx]), requested))
        pressure = 1.0 if self.pump_pressure is None else self.pump_pressure
        cont[self.water_pressure_idx] = np.float32(max(
            float(cont[self.water_pressure_idx]), pressure))
        self.water_sync_steps += 1
        return action
