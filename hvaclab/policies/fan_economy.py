"""Stop paying fan electricity to deliver heat that the coils deliver anyway.

THE PRICE LIST

Per kWh, this reward charges:

    fan / pump electricity      0.0064
    gas heat (boiler, eff ~1)   0.0034
    chiller cooling (COP 3.5)   0.0018   per kWh of *thermal* effect

Electricity costs 1.9x gas per kWh of fuel, and a fan has no COP to divide by
where a chiller has 3.5. So moving air is the most expensive thing this plant
can do, and it is the one the shipped baselines scale *up* with demand.

WHAT ``pid`` DOES, AND WHY IT IS BACKWARDS IN HEATING

Every policy in the chain sets static pressure from the control effort::

    continuous[pressure_idx] = clip(2.0 * abs(u) - 1.0, -1.0, 1.0)

so an air handler working hard commands a hard fan. ``pid``'s own docstring
argues for this against pinning pressure at maximum, and against that baseline
it is an improvement. But ``abs(u)`` does not distinguish the two directions,
and the directions are not symmetric.

**Heating.** The air handler cannot heat -- it only cools, and all heat in this
building comes from the VAV reheat coils fed by the hot-water loop. In cold
weather the supply-air temperature action is inert as well: the handler
delivers ``min(mixed_air, setpoint)``, and with recirculation capped at 0.70,
return air near 293 K and ambient at 231 K, the mixed air is

    0.70 * 293 + 0.30 * 231 = 274 K

which is already below the 285 K floor the setpoint can command. So every
kelvin of zone heating is reheat, and the air arriving at the coil is 20 K
below the zone. Raising static pressure pushes *more* 274 K air past the coil,
and every cubic metre of it has to be heated from 274 K before it does the
zone any good. The fan work is spent, and then gas is spent undoing the air it
moved.

The thermal energy the zone needs is fixed by its heat loss. It can be
delivered as a lot of slightly warm air or a little hot air, and the second is
strictly cheaper here, because the gas is the same either way and the fan is
not. So in heating the fan belongs at its floor and the heat belongs in the
water temperature and the valve position.

**Cooling.** The opposite, and air is the only delivery mechanism, so flow has
to be bought. But the trade still favours temperature over flow: colder supply
air carries more cooling per cubic metre, and the chiller kilowatt-hour is
cheaper than the fan kilowatt-hour by more than three to one. ``pid`` already
drives the supply-air setpoint to its 285 K floor at full cooling effort, so
the correction on this side is smaller -- the fan can come down somewhat
because the air is already as cold as it is allowed to be.

WHAT THIS CHANGES

The pressure command is rewritten after the inherited law has run, which keeps
this a one-line conceptual change against a controller that is otherwise
identical. Three regimes, read off the supply-air temperature action the
inherited law just wrote:

    heating (u > 0)   pressure to ``heating_pressure``, by default the floor
    cooling (u < 0)   pressure scaled by ``cooling_pressure_gain`` < 1
    idle              untouched: the inherited law already drops it to -1
                      and clears the run command

The simulator holds a minimum static pressure under any running handler, so
commanding the floor asks for that minimum rather than for no air at all --
ventilation is not something a policy can trade away here, and this does not
try to.

ONE INCONSISTENCY, STATED

``PIDEconomizerPolicy`` computes its outside-air breakeven from
``_commanded_pressure(u)``, the pressure the inherited law was going to write.
Rewriting the pressure afterwards leaves that calculation reading the old
value. The economizer only acts while cooling, where the change is a scaling
rather than a clamp, and the breakeven moves the same direction as the
pressure it is comparing against -- a lower fan makes outside air *more*
attractive, so the stale value is conservative. Left as is rather than
reimplemented, because overriding it would mean copying the whole chain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.policies.deadline_band import DeadlineBandPolicy


class FanEconomyPolicy(DeadlineBandPolicy):
    """``deadline_band``, with a direction-aware static pressure law.

    Args:
        heating_pressure: Normalised static pressure to command while an air
            handler is heating. -1.0 is the floor the simulator enforces under
            a running handler. ``None`` leaves the inherited ``2|u|-1`` law
            alone, which is the ablation control for the heating side.
        cooling_pressure_gain: Scales the inherited cooling pressure towards
            the floor. 1.0 is the identity, so it is the control for the
            cooling side.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        heating_pressure: float | None = -1.0,
        cooling_pressure_gain: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.heating_pressure = (
            None if heating_pressure is None else float(heating_pressure)
        )
        self.cooling_pressure_gain = float(cooling_pressure_gain)

    @property
    def name(self) -> str:
        return "fan_economy"

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        action = super().get_action(obs)
        continuous = action["continuous_action"]
        discrete = action["discrete_action"]

        for ctrl in self.ahu_controllers:
            pressure_idx = ctrl.get("pressure_idx")
            if pressure_idx is None:
                continue
            # An idle handler has already been given -1 and a cleared run
            # command by the inherited law. Leave it alone: rewriting it would
            # only risk putting a pressure under a handler that is off.
            if not discrete[ctrl["run_discrete_idx"]]:
                continue

            # The supply-air temperature action is the control effort the
            # inherited law just wrote, so its sign is the direction without
            # having to recompute the loop.
            effort = float(continuous[ctrl["temp_idx"]])

            if effort > 0.0:
                if self.heating_pressure is not None:
                    continuous[pressure_idx] = np.float32(self.heating_pressure)
            elif effort < 0.0:
                current = float(continuous[pressure_idx])
                # Scale towards the floor rather than towards zero: zero is the
                # middle of the range, not the minimum.
                scaled = -1.0 + (current + 1.0) * self.cooling_pressure_gain
                continuous[pressure_idx] = np.float32(
                    np.clip(scaled, -1.0, 1.0)
                )

        return action
