"""Open a zone's damper only when the air arriving can help that zone.

THE FLAW

One air handler serves up to ten zones and picks a single supply-air
temperature for all of them. The per-zone damper and reheat valve are the only
place a per-zone decision exists at all -- the starter kit says as much, and
calls it where the headroom over the baselines mostly is.

Every policy in the shipped chain drives the damper from the *magnitude* of the
zone's excursion::

    damper = min(1.0, abs(zone_error) / zone_damper_full_k)

``abs`` is the problem. It asks how far the zone is from its band and never
asks which way, or whether the air about to be delivered is on the useful side
of the zone's own temperature. So when a handler is heating for the sake of its
average zone, a zone that is already too hot opens its damper in proportion to
how hot it is, and receives more warm air. The control action and the error it
is correcting have opposite signs.

This is the per-zone form of the mispricing in ``fan_economy``: an actuator
driven by ``abs(error)``, in a plant where the two directions are not
symmetric.

WHAT CHANGES

Only the case that is unambiguously backwards. A zone above its band is given
airflow only while the supply air is actually colder than the zone; otherwise
its damper goes to the floor, which the simulator holds at 10% open for
ventilation.

The opposite case is left alone deliberately. A zone *below* its band with cold
supply air still wants airflow, because the reheat coil is downstream of the
damper and the heat it adds is carried by that air: shutting the damper would
shut off the zone's only heat source. Air that arrives cold and leaves the coil
hot is doing its job, even though the air itself started on the wrong side.

Supply air temperature is read per handler from
``ahs_<id>_supply_air_temperature_sensor`` rather than inferred from the
commanded setpoint, because the handler delivers
``min(mixed_air, setpoint)`` and in cold weather that is not the setpoint at
all -- the command goes inert below about -6.8 C ambient. The sensor reports
what actually arrives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.policies.fan_economy import FanEconomyPolicy


class ZoneAwarePolicy(FanEconomyPolicy):
    """``fan_economy``, plus a damper that checks the sign of what it delivers.

    Args:
        require_useful_air: When true, a zone above its band keeps its damper
            at the floor unless the supply air is colder than the zone.
            False reproduces the inherited behaviour exactly, which makes it
            the ablation control.
        useful_air_margin_k: How much colder the supply air must be before it
            counts as useful. A small margin avoids opening the damper for a
            fraction of a kelvin, where the fan costs more than the cooling is
            worth -- fan electricity being the dearest fuel in this reward.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        require_useful_air: bool = True,
        useful_air_margin_k: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.require_useful_air = bool(require_useful_air)
        self.useful_air_margin_k = float(useful_air_margin_k)

        names = list(self.field_names)
        norm = self.observation_normalization_metadata.get(
            "supply_air_temperature_sensor", {}
        )
        self._sat_mean = float(norm.get("sample_mean", 0.0))
        self._sat_std = float(norm.get("sample_std", 1.0))

        for ctrl in self.ahu_controllers:
            field = f"ahs_{ctrl['ahu_id']}_supply_air_temperature_sensor"
            ctrl["sat_idx"] = names.index(field) if field in names else None

        # Diagnostics: how often the gate actually fires. A change that never
        # fires and still moves the score would mean something else moved.
        self.gate_closures = 0
        self.zone_commands = 0

    @property
    def name(self) -> str:
        return "zone_aware"

    def _supply_air_temp(self, obs: np.ndarray, ctrl) -> float | None:
        idx = ctrl.get("sat_idx")
        if idx is None:
            return None
        value = float(obs[idx]) * self._sat_std + self._sat_mean
        return value if np.isfinite(value) else None

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        action = super().get_action(obs)
        if not self.require_useful_air:
            return action

        continuous = action["continuous_action"]
        discrete = action["discrete_action"]
        low, high = self._comfort_band

        for ctrl in self.ahu_controllers:
            # A stopped handler moves no air, so its supply-air sensor is
            # reporting a stale reading rather than what any zone is about to
            # receive. Gating on it would be deciding from noise.
            if not discrete[ctrl["run_discrete_idx"]]:
                continue

            supply = self._supply_air_temp(obs, ctrl)
            if supply is None:
                continue

            for _vav_id, obs_idx, action_indices in ctrl["zones"]:
                damper_idx = action_indices.get(
                    "supply_air_damper_percentage_command"
                )
                if damper_idx is None:
                    continue
                self.zone_commands += 1

                zone_temp = self._denormalize_zone_temp(obs[obs_idx])
                if zone_temp <= high:
                    continue  # not above its band; nothing to second-guess

                # Above the band, so this zone wants cooling. Airflow only
                # delivers cooling while the air is colder than the zone.
                if supply >= zone_temp - self.useful_air_margin_k:
                    continuous[damper_idx] = np.float32(-1.0)
                    self.gate_closures += 1

        return action
