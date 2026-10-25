"""Occupancy-gated control: stop paying for comfort nobody is there to receive.

THE OBSERVATION THIS IS BUILT ON

The comfort term of the reward is occupancy-weighted. From
``base_setpoint_energy_carbon_reward``::

    if total_occupancy > 0.0:
        normalized_productivity_regret = actual/max - 1.0
    else:
        normalized_productivity_regret = 0.0

So on any step where the building is empty the comfort term is *identically
zero*, whatever the zone temperatures are, and the step's reward is nothing but

    -0.2 * energy - 0.2 * carbon

Occupants arrive between hour 6 and 12 and leave between 13 and 19, and
``is_work_day`` forces every occupant AWAY on weekends and holidays. Over a
30-day strip that leaves roughly 60% of the 8,640 scored steps with nobody in
the building. Every joule the plant burns in those steps buys exactly nothing.

Measured on ``building_train_1/fairbanks_1``, the scenario where the shipped
baselines do worst, the weighted split of ``pid_water``'s loss is

    comfort 0.087   energy 0.077   carbon 0.122

so 69% of the loss is energy and carbon, and the largest single term is carbon
from the gas boiler. The shipped baselines spend that because they hold the eco
band (289-298 K) around the clock: in Fairbanks that is a 58 K gradient to
outside, maintained all night for no comfort credit at all.

WHAT THIS POLICY CHANGES

One decision, taken per step: is comfort being scored right now, and if not,
can the building still reach its band by the time it is? Three regimes follow.

  ACTIVE     Somebody is in the building. Comfort is being scored, and it is
             worth far more than energy -- see the arithmetic below -- so the
             plant runs on the inherited `pid_water` control law.

  RECOVER    Empty, but the building can no longer afford to drift further and
             still be in band at the deadline. Same control law, driven to the
             *comfort* band rather than the eco band.

  COAST      Empty, and still recoverable. Everything off.

WHY COMFORT OUTRANKS ENERGY WHILE OCCUPIED

The regret is a weighted mean of each zone's shortfall, and shortfall is
``deviation / 5 K`` inside the knee. So every occupied zone sitting 1 K outside
its band costs ``0.6 * 0.2 = 0.12`` of reward per step. One kWh of electricity
costs ``0.2*(cost/6.37) + 0.2*(carbon/15.14) = 0.0064``. Closing 1 K of
building-wide deviation is therefore worth about 19 kWh per five-minute step,
or 225 kW held continuously -- 70% of the 320 kW the whole plant can draw.
While the building is occupied, energy is close to free next to comfort. While
it is empty, comfort is free and energy is the entire bill.

That asymmetry is the whole policy.

THE DEADLINE, AND WHY THE FLOOR MOVES

A policy that simply switched everything off would arrive at 06:00 with the
building cold, and the morning's comfort penalty would dwarf the night's
saving -- the quadratic branch past 5 K is unbounded. So COAST is bounded by
feasibility rather than by a fixed setback temperature:

    recoverable = heating_rate * steps_until_occupancy * SAFETY
    floor       = comfort_low - recoverable

``floor`` is the coldest the building may be *right now* and still reach its
band in time. Early in the night ``steps_until_occupancy`` is large, so the
floor is far below anything the weather will produce and the plant stays off.
As the deadline approaches the floor rises to meet the actual zone
temperature, and the moment it crosses, RECOVER begins. The ramp is not
scheduled anywhere; it falls out of the feasibility test.

``heating_rate`` is not a constant and is not fitted offline. It is measured
from this building's own response, which is what lets the same code transfer to
a building whose envelope, zone count and air-handler layout were never seen.
It starts deliberately pessimistic, so the first night preconditions far too
early and the estimate is corrected upward from evidence rather than guessed
downward from optimism. Being early costs a few hours of gas once; being late
costs an unbounded comfort penalty every morning.

``comfort_mode_soon`` is not used for this. It looks exactly 60 minutes ahead,
and in deep cold a building can need far longer than an hour to recover, so it
answers a question this controller is not asking.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

# How much of the measured heating rate to trust when deciding the floor.
# Below 1.0 the controller assumes it recovers more slowly than it has been
# seen to, so it starts earlier than strictly necessary. Recovery is slower
# when it matters most -- the coldest nights are also the ones with the least
# margin -- so the margin is taken here rather than hoped for.
RECOVERY_SAFETY = 0.5

# Rate estimates, in K per 5-minute step, used until the building has shown
# what it can actually do. 0.004 K/step is 0.05 K/h, far below any real
# plant, which makes the first night's lead time long on purpose.
INITIAL_HEAT_RATE_K_PER_STEP = 0.004
INITIAL_COOL_RATE_K_PER_STEP = 0.004

# Floor on the rate estimate, so a night where the building happened not to
# move cannot drive the lead time to infinity and pin the plant on.
MIN_RATE_K_PER_STEP = 0.002
# Ceiling, so one noisy step cannot convince the controller it recovers
# instantly and let it start far too late.
MAX_RATE_K_PER_STEP = 0.20

# Smoothing on the rate estimate. Rates are updated towards the observed value
# only when the plant is actually driving hard, so samples are sparse and each
# one carries real information; a slow filter would waste them.
RATE_EMA = 0.1

# The control effort above which a step counts as evidence about how fast this
# building can be driven. Below it the zone is drifting, not recovering.
EVIDENCE_EFFORT = 0.6

STEPS_PER_HOUR = 12
HOURS_PER_DAY = 24
STEPS_PER_DAY = STEPS_PER_HOUR * HOURS_PER_DAY


class OccupancyGatedPolicy(PIDWaterResetPolicy):
    """``pid_water``, gated on whether comfort is being scored at all.

    The inherited control law is left untouched. Only the decision of *when* to
    run it is new, which keeps the comparison against ``pid_water`` a clean
    one-factor ablation.
    """

    @property
    def name(self) -> str:
        return "occ"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        names = list(self.field_names)
        self.occupancy_idx = names.index("num_occupants")

        # `comfort_mode_now` is the environment's own
        # `SetpointSchedule.is_comfort_mode(t)`, which tests the hour window,
        # the holiday list AND the weekend. It is therefore strictly better
        # than reconstructing the calendar from the cyclic hour-of-day and
        # day-of-week features, which cannot see holidays at all.
        #
        # `comfort_mode_soon` is the same predicate 60 minutes ahead.
        self.comfort_now_idx = names.index("comfort_mode_now")
        self.comfort_soon_idx = names.index("comfort_mode_soon")

        # `num_occupants` is documented as normalised by
        # `occupancy_normalization_constant`, but the environment default is
        # 0.0 and `evaluate.py` assigns the *private* attribute, which the
        # observation builder never reads. So the feature arrives as a raw
        # count. Rather than rely on that, treat anything at or below the
        # lowest value seen so far as empty: an empty building is by far the
        # most common state, so the running minimum is the zero level under
        # either convention.
        self._occupancy_zero_level = 0.0

        self._heat_rate = INITIAL_HEAT_RATE_K_PER_STEP
        self._cool_rate = INITIAL_COOL_RATE_K_PER_STEP
        self._prev_worst_cold: float | None = None
        self._prev_worst_hot: float | None = None
        self._prev_mode = "COAST"

        # Diagnostics, read by the analysis scripts rather than by the control
        # law. Cheap to keep and the only way to see why a night went wrong.
        self.mode_counts = {"ACTIVE": 0, "RECOVER": 0, "COAST": 0}

    # ---------------------------------------------------------------- helpers

    def _is_empty(self, obs: np.ndarray) -> bool:
        """Is the building unoccupied, so that comfort is not being scored?

        The reward gates on ``total_occupancy > 0.0`` over the same interval
        this observation reports, so this is the exact switch the scorer uses.
        """
        value = float(obs[self.occupancy_idx])
        if value < self._occupancy_zero_level:
            self._occupancy_zero_level = value
        return value <= self._occupancy_zero_level + 1e-6

    def _steps_until_comfort_window(self, hour: int, day: int) -> int:
        """Steps until the comfort window next opens.

        Only called when the window is shut, so "later today" means today's
        opening has not happened yet.

        The cyclic hour-of-day and day-of-week features are the only calendar
        signal in the observation, so holidays are invisible here and a holiday
        gets preconditioned as if it were a work day. That is the right way to
        be wrong: one morning of gas against a comfort penalty that would have
        been unbounded had the guess gone the other way. ``comfort_mode_now``
        does see holidays, so the mistake only ever costs preconditioning, and
        never conditioning during a day that is actually scored.
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

    def _worst_deviations(self, obs: np.ndarray) -> tuple[float, float]:
        """How far the worst zone is below, and above, the *comfort* band.

        Both are returned because a single building can need heating in one
        wing and cooling in another, and the floor test has to respect
        whichever is tighter.
        """
        low, high = self.comfort_temp_window
        worst_cold = 0.0
        worst_hot = 0.0
        for ctrl in self.ahu_controllers:
            for vav_id, obs_idx, _ in ctrl["zones"]:
                # The comfort band, not the zone's current sensor band: at
                # night the sensors report the eco band, and the question here
                # is about the band that will be scored at the deadline.
                temp = self._denormalize_zone_temp(obs[obs_idx])
                worst_cold = max(worst_cold, low - temp)
                worst_hot = max(worst_hot, temp - high)
        return worst_cold, worst_hot

    def _update_rates(self, worst_cold: float, worst_hot: float) -> None:
        """Learn how fast this building actually recovers.

        Only steps where the plant was driving hard count, because a step spent
        drifting says nothing about capability. The signal is the reduction in
        the worst zone's deviation, which is the quantity the floor test needs
        to predict.
        """
        efforts = getattr(self, "_last_efforts", None) or []
        if not efforts:
            self._prev_worst_cold = worst_cold
            self._prev_worst_hot = worst_hot
            return

        heating_hard = max((u for u in efforts), default=0.0) >= EVIDENCE_EFFORT
        cooling_hard = min((u for u in efforts), default=0.0) <= -EVIDENCE_EFFORT

        if heating_hard and self._prev_worst_cold is not None:
            gained = self._prev_worst_cold - worst_cold
            if gained > 0.0:
                self._heat_rate = float(np.clip(
                    (1 - RATE_EMA) * self._heat_rate + RATE_EMA * gained,
                    MIN_RATE_K_PER_STEP, MAX_RATE_K_PER_STEP))

        if cooling_hard and self._prev_worst_hot is not None:
            gained = self._prev_worst_hot - worst_hot
            if gained > 0.0:
                self._cool_rate = float(np.clip(
                    (1 - RATE_EMA) * self._cool_rate + RATE_EMA * gained,
                    MIN_RATE_K_PER_STEP, MAX_RATE_K_PER_STEP))

        self._prev_worst_cold = worst_cold
        self._prev_worst_hot = worst_hot

    def _off_action(self) -> dict[str, np.ndarray]:
        """Everything stopped.

        Zero is the middle of each normalised range, not the off position, so
        every slot is written explicitly. Dampers and valves go to -1 (shut,
        which the simulator will floor at 10% open for any handler still
        running), the water loop to its minimum temperature and pressure, and
        every discrete run command to 0.

        Recirculation is commanded to the cap. With the fan stopped it should
        not matter, but if anything downstream still moves air it should be
        moving air already inside the building rather than outside air at the
        ambient temperature.
        """
        continuous = -np.ones(self.n_continuous, dtype=np.float32)
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        return {"continuous_action": continuous, "discrete_action": discrete}

    # ------------------------------------------------------------------ mode

    def _select_mode(self, obs: np.ndarray) -> str:
        """ACTIVE, RECOVER or COAST for this step."""
        if not self._is_empty(obs):
            return "ACTIVE"

        # Empty, but inside the scored window. This is NOT a coasting
        # opportunity, and treating it as one is the single most expensive
        # mistake available here.
        #
        # Occupants arrive on independent stochastic draws from hour 6 to 12,
        # so the building is routinely empty for a few steps *inside* the
        # window -- and the regret is normalized by TOTAL occupancy, so the
        # first person to walk in carries the entire comfort score for their
        # zone alone. A controller that switches off because the building
        # happens to be empty at 06:05 is off when that person arrives, with
        # every zone out of band and no averaging to hide it.
        #
        # The saving was never in these hours anyway. The comfort window is
        # 13 h x 5 days = 39% of the week; the other 61% is nights, weekends
        # and holidays, and that is where COAST earns its keep.
        if float(obs[self.comfort_now_idx]) > 0.5:
            return "RECOVER"

        # Within 60 minutes of the window opening. Hold the band from here
        # regardless of what the feasibility test thinks, because the test is
        # built on an estimated rate and this is where a wrong estimate stops
        # being recoverable.
        if float(obs[self.comfort_soon_idx]) > 0.5:
            return "RECOVER"

        hour, day = self._get_hour_and_day(obs)
        steps_left = self._steps_until_comfort_window(hour, day)
        worst_cold, worst_hot = self._worst_deviations(obs)

        # How much deviation can still be closed before the deadline.
        recoverable_cold = self._heat_rate * steps_left * RECOVERY_SAFETY
        recoverable_hot = self._cool_rate * steps_left * RECOVERY_SAFETY

        if worst_cold > recoverable_cold or worst_hot > recoverable_hot:
            return "RECOVER"
        return "COAST"

    # ---------------------------------------------------------------- action

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        worst_cold, worst_hot = self._worst_deviations(obs)
        self._update_rates(worst_cold, worst_hot)

        mode = self._select_mode(obs)
        self.mode_counts[mode] += 1
        self._prev_mode = mode

        if mode == "COAST":
            # No comfort is being scored and the building can still make its
            # deadline, so the cheapest valid action is the right one.
            self._last_efforts = []
            return self._off_action()

        if mode == "ACTIVE":
            return super().get_action(obs)

        # RECOVER. The inherited law reads the band from the schedule, which
        # outside comfort hours is the wide eco band -- and driving to the eco
        # band is exactly what leaves the building short at 06:00. Narrow the
        # eco window to the comfort window for this call so the same control
        # law aims at the band that will actually be scored.
        saved = self.eco_temp_window
        self.eco_temp_window = self.comfort_temp_window
        try:
            return super().get_action(obs)
        finally:
            self.eco_temp_window = saved
