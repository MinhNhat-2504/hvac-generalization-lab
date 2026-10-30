"""Give the reheat coils the airflow they need, and no more.

THE DIAGNOSIS

After the fan clamp, the per-zone PI and the cooling-run hold, comfort loss
across the 30 published scenarios correlates $-0.78$ with mean outdoor
temperature. Binned:

    below 0 C      comfort 0.0346   energy+carbon 0.0312
    0 to 15 C      comfort 0.0286   energy+carbon 0.0073
    15 to 25 C     comfort 0.0046   energy+carbon 0.0061
    above 25 C     comfort 0.0030   energy+carbon 0.0134

Comfort is ten times worse below 15 C than above it, and the 0-15 C band is
almost as bad as the deep cold while spending a quarter of the energy. So this
is not a deep-cold problem and not an energy problem. **It is a heating-mode
comfort problem**, and the worst single scenario by comfort,
``fairbanks_7``, has a mean of +6.4 C and spends no hours below -6.8 C at all.

THE MECHANISM

All heat in this building comes from the VAV reheat coils, and a coil heats
the air passing through it. No airflow, no heat delivered -- the valve can be
wide open and the zone still gets nothing.

But the decision to *run* an air handler is still the inherited loop's, and it
makes that decision on the handler's **aggregate** zone error against the
band. When the aggregate is inside the band the handler idles, and every zone
under it loses its airflow at once, including the cold outlier the per-zone
loop is trying to rescue. The per-zone PI can ask for heat; it cannot create
the air to carry it.

We had already found this for cooling and fixed it there, by forcing handlers
on. But that fix was gated on the zones asking to be *cooled*, because forcing
every handler on cost 54% in Fairbanks. So the heating side kept the bug.

WHY THE COOLING FIX COST SO MUCH IN THE COLD, AND WHAT IS DIFFERENT HERE

The cooling version forced a moderate supply temperature and a mid-range
static pressure. In the cold that is the worst of both: the supply-air command
is inert below about -6.8 C ambient, so it buys nothing, while the pressure
pushes 274 K air through the building at three tenths ambient -- the load the
fan clamp exists to avoid.

This asks for the opposite on both axes:

    supply air temperature  +1.0   the warmest command available, which means
                                   "do not cool" -- the handler can only cool,
                                   so commanding 305 K leaves the mixed air
                                   alone rather than chilling it further
    static pressure         -1.0   the floor, so the coils get the minimum
                                   available airflow; whether it carries
                                   enough heat has to be measured

The efficiency argument is the same one the fan clamp rests on. Heat reaching
a zone is ``flow * cp * (T_coil_out - T_zone)``, while heat the coil must add
is ``flow * cp * (T_coil_out - T_supply)``. With the zone at 294 K and supply
at 274 K the delivered fraction is ``(T_coil_out - 294)/(T_coil_out - 274)``:
23% at 300 K and 41% at 308.15 K, the simulator's default discharge limit.
Hotter air can deliver heat with less ambient air dragged in, but the discharge
limit also caps heating capacity at a given flow. Minimum flow is therefore an
efficiency hypothesis, not a guarantee that every zone receives enough heat.

EXPERIMENT BOUNDARY

This is an airflow-only intervention: hot-water commands have already been
computed by the inherited policy and are left unchanged. If that policy idled
every handler, it also stopped the water pump. Restarting air alone then
delivers no reheat. The diagnostic probe measures how often this occurs; see
reports/HEAT_DELIVERY_REVIEW.md for the completed ablation and its limitations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.policies.cooling_hold import CoolingHoldPolicy


class HeatDeliveryPolicy(CoolingHoldPolicy):
    """``cooling_hold``, plus minimum airflow for handlers whose zones want heat.

    Args:
        heat_airflow: Run an idle handler at minimum static pressure when its
            zones are asking for heat, so the reheat coils have something to
            heat. False is the ablation control.
        heat_pressure: Static pressure for such a handler. -1.0 is the floor;
            the simulator holds a minimum under any running handler.
        heat_sat: Supply-air temperature command for such a handler. +1.0 is
            the warmest available, which means "do not cool" -- the handler
            cannot add heat, so this leaves the mixed air alone.
        heat_demand_frac: Fraction of a handler's zones that must be asking
            for heat before it is run. A single outlier in a large building is
            often unreachable, and starting a handler for it spends airflow on
            every other zone too.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        heat_airflow: bool = True,
        heat_pressure: float = -1.0,
        heat_sat: float = 1.0,
        heat_demand_frac: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.heat_airflow = bool(heat_airflow)
        self.heat_pressure = float(heat_pressure)
        self.heat_sat = float(heat_sat)
        self.heat_demand_frac = float(heat_demand_frac)
        self.heat_starts = 0

    @property
    def name(self) -> str:
        return "heat_delivery"

    def _allow_heat_override(self, obs, ctrl, demands) -> bool:
        """Extension point; the original airflow-only experiment has no guard."""
        return True

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        action = super().get_action(obs)
        if not (self.heat_airflow and self._pi_active):
            return action

        continuous = action["continuous_action"]
        discrete = action["discrete_action"]

        for ctrl in self.ahu_controllers:
            if not ctrl["zones"]:
                continue
            # Already running, for whatever reason: leave it.
            if discrete[ctrl["run_discrete_idx"]]:
                continue

            # `_zone_u` is written by the per-zone PI on this same step.
            # Negative is a zone asking to be warmed.
            demands = [self._zone_u.get(v, 0.0) for v, _, _ in ctrl["zones"]]
            wanting_heat = sum(1 for u in demands if u < 0.0)
            if not wanting_heat:
                continue
            if wanting_heat / len(demands) < self.heat_demand_frac:
                continue
            if not self._allow_heat_override(obs, ctrl, demands):
                continue

            discrete[ctrl["run_discrete_idx"]] = 1
            continuous[ctrl["temp_idx"]] = np.float32(self.heat_sat)
            if ctrl.get("pressure_idx") is not None:
                continuous[ctrl["pressure_idx"]] = np.float32(self.heat_pressure)
            # Hold the heat in. Recirculated air is the warmest available and
            # the handler cannot add any of its own.
            if ctrl.get("recirc_idx") is not None:
                continuous[ctrl["recirc_idx"]] = np.float32(1.0)
            # Count handler command overrides, not physical OFF-to-ON edges.
            # HWS is intentionally unchanged in this airflow-only experiment.
            self.heat_starts += 1

        return action
