"""Deadline-aware comfort band: widen it when nobody is being scored.

WHAT THE REWARD ACTUALLY GATES ON

The comfort term is occupancy-weighted, and the weighting is a hard gate. From
``base_setpoint_energy_carbon_reward``::

    if total_occupancy > 0.0:
        normalized_productivity_regret = actual/max - 1.0
    else:
        normalized_productivity_regret = 0.0

On any step with nobody in the building the comfort term is identically zero
whatever the zone temperatures are, and the step bills only
``-0.2*energy - 0.2*carbon``. Occupants arrive from hour 6 to 12, leave from 13
to 19, and ``is_work_day`` sends everyone away on weekends and holidays, so
roughly 60% of the 8,640 scored steps are unoccupied.

WHAT THE FIRST ATTEMPT GOT WRONG

The obvious move -- switch the plant off when the building is empty -- was
tried in ``occupancy_gated.py`` and measured *worse* than ``pid_water`` on all
three reference scenarios. Two separate reasons, both worth recording because
each one on its own would have been invisible:

1. **Being empty inside the scored window is not an opportunity.** Arrivals are
   independent stochastic draws, so the building is routinely empty for a few
   steps *after* hour 6 -- and the regret is normalized by TOTAL occupancy, so
   the first person through the door carries their zone's entire comfort score
   alone. Switching off at 06:05 because nobody has arrived yet means being off,
   with every zone out of band, when they do.

2. **Forcing the comfort band outside the window costs more than coasting
   saves.** The first attempt drove to the *comfort* band (294-297 K) whenever
   it judged recovery necessary, where the shipped baselines use the wide eco
   band (289-298 K). In Fairbanks that is 5 K of extra heating all night, and
   it showed up exactly where it should: unoccupied-step loss went **up**, from
   447 to 830, against a comfort gain of only 0.010.

So the lever is not an on/off gate. It is the **band itself**.

HOW THIS WORKS INSTEAD

There are no modes. One quantity is computed per step -- the band this
controller should be holding right now -- and handed to the inherited control
law:

    inside the scored window      the comfort band, exactly
    outside it, far from it       comfort band widened by the full setback
    outside it, approaching it    widened by a fraction, ramping linearly to
                                  the comfort band at the deadline

The inherited law already does the right thing with a wide band: when every
zone is inside it, ``pid`` idles its air handler, shuts the damper, closes the
reheat valve, drops the water loop to its minimum and clears every discrete run
command. A band wide enough to contain the building *is* the off action, and it
reaches it through the same code path that handles everything else, so there is
no separate shutdown branch to get wrong.

The ramp is what the first attempt was missing. Holding a wide band until the
deadline and then demanding the comfort band would need an impossible recovery;
narrowing it linearly over ``lead_steps`` asks the plant for a gradual approach
it can actually deliver, and asks for nothing at all before that.

WHY THE SETBACK IS BOUNDED, AND NOT DERIVED

A tempting formulation makes the setback depth a feasibility calculation:
widen by ``recovery_rate * steps_until_deadline``, so the band is only as wide
as the building can still climb back from. That was the first attempt's design
and the recovery rate proved the weak link -- it was estimated from the *worst
zone's* deviation closing, which overstates what the plant can do for a whole
building at once, because the plant's capacity is shared across zones and the
worst zone changes from step to step.

So the depth is a bounded parameter here, and the evidence says bounded is
right on its own merits: on ``fairbanks_1`` the shipped ``fixed_setpoint``,
which runs **one band around the clock with no setback at all**, scores far
better than ``pid_water``, which does set back. In a climate that cold,
recovering from a setback runs the fan at full static pressure and the water
loop at its maximum -- and fan electricity is the most expensive fuel in this
reward (0.0064 per kWh, against 0.0034 for gas heat and 0.0018 for chiller
cooling at COP 3.5, which has a COP to divide by where a fan does not). Steady
holding can genuinely beat cycling.

Which means the right setback is climate-dependent and has to be measured, not
assumed, so it is exposed as a parameter with variants registered for
ablation rather than fixed at a value that looked reasonable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

STEPS_PER_HOUR = 12
HOURS_PER_DAY = 24


class DeadlineBandPolicy(PIDWaterResetPolicy):
    """``pid_water`` driven by a deadline-aware band instead of the schedule's.

    Args:
        setback_k: How far below the comfort band's lower edge to let zones sit
            while the building is unoccupied and the deadline is far. The
            shipped eco band uses 5 K.
        setup_k: The same above the upper edge. The shipped eco band uses only
            1 K, which is conservative given that chiller cooling is the
            cheapest thermal kilowatt-hour available here.
        lead_hours: Over how long before the window opens the band narrows back
            to the comfort band. This is the recovery time the plant is given.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        setback_k: float = 5.0,
        setup_k: float = 3.0,
        lead_hours: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)

        self.setback_k = float(setback_k)
        self.setup_k = float(setup_k)
        self.lead_steps = max(1.0, float(lead_hours) * STEPS_PER_HOUR)

        names = list(self.field_names)
        # `comfort_mode_now` is the environment's own
        # `SetpointSchedule.is_comfort_mode(t)`, which tests the hour window,
        # the holiday list AND the weekend. Strictly better than rebuilding the
        # calendar from the cyclic time features, which cannot see holidays.
        self.comfort_now_idx = names.index("comfort_mode_now")
        self.occupancy_idx = names.index("num_occupants")

        # The schedule's own windows, kept because `get_action` overwrites the
        # inherited attributes every step and needs the originals to build
        # from. `comfort_temp_window` is the band that is actually scored.
        self._comfort_band = tuple(self.comfort_temp_window)

        # Diagnostics only; the control law does not read these.
        self.band_log: dict[str, int] = {"scored": 0, "setback": 0, "ramp": 0}

    @property
    def name(self) -> str:
        return "deadline_band"

    # ----------------------------------------------------------------- timing

    def _steps_until_window(self, hour: int, day: int) -> int:
        """Steps until the comfort window next opens.

        Only meaningful when the window is shut, which is the only place it is
        called.

        Holidays are invisible here -- the cyclic hour-of-day and day-of-week
        features are the only calendar in the observation -- so a holiday is
        preconditioned as though it were a work day. That is the right way to
        be wrong: it costs one morning of recovery that was not needed, where
        the opposite error costs an unbounded comfort penalty.
        ``comfort_mode_now`` does see holidays, so the mistake never leaks into
        a day that is actually scored.
        """
        start = self.morning_start_hour
        if day < 5 and hour < start:
            return (start - hour) * STEPS_PER_HOUR

        days_ahead = 1
        while days_ahead <= 7:
            if (day + days_ahead) % 7 < 5:
                break
            days_ahead += 1
        return (days_ahead * HOURS_PER_DAY - hour + start) * STEPS_PER_HOUR

    # ------------------------------------------------------------------- band

    def _target_band(self, obs: np.ndarray) -> tuple[float, float]:
        """The band to hold this step."""
        low, high = self._comfort_band

        # Inside the scored window: hold the band that is being scored, and do
        # not ask whether anyone has arrived yet. See reason 1 in the module
        # docstring -- this is where the first attempt lost most of its score.
        if float(obs[self.comfort_now_idx]) > 0.5:
            self.band_log["scored"] += 1
            return low, high

        hour, day = self._get_hour_and_day(obs)
        steps_left = self._steps_until_window(hour, day)

        # Linear narrowing over the last `lead_steps`, full setback before it.
        # At steps_left == 0 this returns the comfort band exactly, so the band
        # is continuous across the window opening and the controller is not
        # handed a step change it has to chase.
        fraction = min(1.0, steps_left / self.lead_steps)
        if fraction >= 1.0:
            self.band_log["setback"] += 1
        else:
            self.band_log["ramp"] += 1

        return low - self.setback_k * fraction, high + self.setup_k * fraction

    # ----------------------------------------------------------------- action

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        band = self._target_band(obs)

        # Both windows are overwritten, not just the eco one. The inherited
        # `get_action` chooses between them by rebuilding the work-hour test
        # from the cyclic time features, which cannot see holidays -- so on a
        # holiday it would take the comfort branch while `comfort_mode_now` is
        # false. Writing the same band into both makes the choice irrelevant
        # and keeps this class the only thing deciding the setpoints.
        self.comfort_temp_window = band
        self.eco_temp_window = band
        return super().get_action(obs)
