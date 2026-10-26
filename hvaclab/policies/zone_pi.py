"""Per-zone PI control, aimed at the band centre instead of its edge.

WHY THE SHIPPED ZONE LOOP OSCILLATES

Every policy in the starter kit chain commands a zone like this::

    zone_error = deadband_error(zone_temp, low, high)      # 0 inside the band
    damper     = min(1.0, abs(zone_error) / 2.0)

Two properties follow, and together they are the whole problem.

**It is proportional only, and it aims at the edge.** ``deadband_error`` is
zero anywhere inside the band, so the loop has no target inside it -- it pushes
until the zone crosses the edge and then stops pushing. A proportional law with
no integral term cannot hold a position against a standing load anyway; it
settles wherever its gain and the load balance, which here is somewhere outside
the band.

**And its gain is high.** Full damper at 2 K of excursion is a hard push for a
zone that is 0.3 K out. The zone crosses the band, the error goes to zero, the
damper shuts, the zone drifts back out the other side.

The public "Probe ZT Best" notebook measured the consequence directly: zones
averaged *inside* the 21-24 C band and yet sat outside it 57-61% of working
hours, swinging 2-4 K every 5-15 minutes, with comfort at 60% of the total
loss. That is a controller oscillating across its target, not one that lacks
authority.

WHAT THIS DOES INSTEAD

Each zone runs its own PI loop against the band *centre*:

    e = T_zone - centre
    u = clip(Kp e + Ki I, -1, 1)
    damper = |u|          the air is needed either way
    reheat = -u           positive e wants cooling, negative wants heat

Aiming at the centre gives the loop somewhere to be, and the integral removes
the steady-state offset that a softer proportional gain would otherwise leave.
The gain is five times softer than the shipped 1/2.0 K: Kp 0.1 with Ki 0.005
per five-minute step.

WHAT WE GOT WRONG BEFORE THIS

This repository previously tested a stiffer loop (kp 0.5 -> 1.0), dampers
opening sooner (2 K -> 1 K -> 0.5 K) and a tighter target band, measured all
three as worse, and concluded that "the shipped gains sit near a local optimum
for this control structure". Every one of those was a push in the direction
that *causes* the oscillation. The correct direction was much softer, and the
reason the obvious softer-P experiment is not enough on its own is the missing
integral term -- soft P alone trades the swing for a standing offset. Neither
half works without the other, which is how an ablation that moves one knob at a
time misses it.

Credit: the diagnosis and the gains are from the public notebook "Smart
Buildings: per-zone PI control and AHU cooling hold" (Probe ZT Best), shared on
the competition's Code tab. The implementation here is our own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.policies.fan_economy import FanEconomyPolicy


class ZonePIPolicy(FanEconomyPolicy):
    """Per-zone PI to the band centre, replacing the shipped proportional law.

    Args:
        kp: Proportional gain per kelvin of deviation from the band centre.
            The shipped loop is effectively 0.5 (full damper at 2 K).
        ki: Integral gain per five-minute step.
        integral_limit: Clamp on the accumulated integral, in kelvin-steps.
        prestart_hours: How long before the comfort window the loops start, so
            the building is in band when occupancy begins rather than starting
            to correct then.
        enabled: False restores the inherited law exactly, which makes it the
            ablation control.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        kp: float = 0.1,
        ki: float = 0.005,
        integral_limit: float = 200.0,
        prestart_hours: float = 1.0,
        enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.zone_kp = float(kp)
        self.zone_ki = float(ki)
        self.zone_integral_limit = float(integral_limit)
        self.prestart_hours = float(prestart_hours)
        self.zone_pi_enabled = bool(enabled)

        self._zone_integral: dict[str, float] = {}
        self._pi_active = False

    @property
    def name(self) -> str:
        return "zone_pi"

    # ------------------------------------------------------------------ gate

    def _in_pi_window(self, obs: np.ndarray) -> bool:
        """Is the per-zone loop running this step?

        From ``prestart_hours`` before the comfort window opens until it
        closes, on work days only. Outside that, comfort is not scored and the
        inherited law's behaviour -- which idles the plant when zones are in
        band -- is what we want.
        """
        if float(obs[self.comfort_now_idx]) > 0.5:
            return True
        hour, day = self._get_hour_and_day(obs)
        if day >= 5:
            return False
        start = self.morning_start_hour - self.prestart_hours
        return start <= hour < self.evening_start_hour

    # --------------------------------------------------------------- command

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        """Per-zone PI, or the inherited proportional law outside the window."""
        if not (self.zone_pi_enabled and self._pi_active):
            return super()._command_zones(continuous, zones, obs, low, high)

        centre = 0.5 * (low + high)

        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue

            error = self._denormalize_zone_temp(obs[obs_idx]) - centre
            integral = self._zone_integral.get(vav_id, 0.0)

            # Conditional integration. The integral only accumulates while the
            # output is off its limits, or while the error would unwind it --
            # otherwise a zone the plant cannot reach winds up for hours and
            # then takes just as long to let go once it finally arrives.
            candidate = integral + error
            unclipped = self.zone_kp * error + self.zone_ki * candidate
            if abs(unclipped) < 1.0 or (unclipped > 0) != (error > 0):
                integral = float(np.clip(
                    candidate,
                    -self.zone_integral_limit,
                    self.zone_integral_limit,
                ))
            self._zone_integral[vav_id] = integral

            u = float(np.clip(
                self.zone_kp * error + self.zone_ki * integral, -1.0, 1.0
            ))

            # The air is needed in both directions: to cool, and to carry the
            # reheat coil's output. So the damper follows the magnitude and
            # only the valve follows the sign.
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = np.float32(2.0 * abs(u) - 1.0)
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                opening = max(0.0, -u)
                continuous[reheat_idx] = np.float32(2.0 * opening - 1.0)

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        # Decided once per step and read by `_command_zones`, which the
        # inherited `get_action` calls per air handler and cannot be passed
        # extra arguments without copying the whole method.
        self._pi_active = self._in_pi_window(obs)
        if not self._pi_active:
            self._zone_integral.clear()
        return super().get_action(obs)
