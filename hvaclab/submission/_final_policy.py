"""The controller this entry submits, with no dependency on this repository.

Concatenated after the starter kit's ``pid.py`` and ``pid_plus.py`` by
``hvaclab.submission.build``, so ``PIDWaterResetPolicy`` is already defined
above it in the assembled file. The import below exists only so this file is
readable and type-checkable on its own; the build strips it.

The equivalence to what was measured is asserted by ``build --verify``, which
rolls this class and the registry variant it claims to reproduce through the
same scenario and requires the returns to match exactly. The simulator is
deterministic, so anything short of an exact match means the submitted
controller is not the measured one.

CREDIT

The per-zone PI diagnosis and gains, and the cooling-run hold, are from the
public notebook "Smart Buildings: per-zone PI control and AHU cooling hold"
(Probe ZT Best) shared on the competition's Code tab. The implementations here
are our own, as is the heating-side fan clamp -- which that notebook reports
independently, with the same explanation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.submission._shared import PIDWaterResetPolicy

#: The registry key this class must reproduce exactly.
EQUIVALENT_TO = "h_288"

STEPS_PER_HOUR = 12


class FinalPolicy(PIDWaterResetPolicy):
    """``pid_water`` with four changes, none of them learned or fitted.

    THE ORGANISING FACT

    One asymmetry decides three of the four. **In heating, airflow is the
    load. In cooling, it is the delivery.** The air handler cannot heat -- all
    heat comes from the VAV reheat coils -- and below about -6.8 C ambient its
    supply-air action is inert too, because it delivers
    ``min(mixed_air, setpoint)`` and with recirculation capped at 0.70, return
    air near 293 K and ambient at 231 K the mixed air is already 274 K, below
    the 285 K floor the setpoint can command.

    So in the cold, every extra cubic metre the fan moves is three tenths
    ambient air that must then be heated: the fan work is spent, and then gas
    is spent undoing the air the fan moved. In the heat the same airflow is the
    only way cooling reaches a zone. Every change below that adds or withholds
    airflow therefore needs the same gate, and each one was measured to pay in
    one direction and cost in the other.

    THE CHANGES

    1. **The band never sets back.** Holding the comfort band around the clock
       beat every setback depth tried. Not an energy argument: a building held
       in band starts the morning with no deviation to recover, and the
       recovery costs more than the setback saves.

    2. **Static pressure goes to its floor while heating.** The clamp above.
       Cooling is left exactly as ``pid`` wrote it, because there the airflow
       is the delivery.

    3. **Per-zone PI against the band centre.** The shipped zone law is
       proportional only and aims at the band *edge* -- ``deadband_error`` is
       zero inside the band, so there is no target in there to hold -- and its
       gain is high enough to drive a zone 0.3 K out straight through to the
       other side. Measured elsewhere: zones averaged inside the 21-24 C band
       and yet sat outside it 57-61% of working hours. A five-times softer
       gain with an integral term fixes both halves; neither half works alone,
       since soft P trades the swing for a standing offset and an integral on
       a deadband error has nothing inside the band to integrate towards.

    4. **A minimum run length for cooling runs.** The shipped loop idles a
       handler the instant its aggregate zone error clears and restarts it at
       the next excursion -- 50 to 107 switches per handler per day. Holding a
       run that was started to cool, at its last commands, removes most of the
       hot-climate loss. Heating runs are not held: a handler that cannot heat
       has no reason to stay on.

    Measured on the 30 published scenarios, full 8,640-step episodes:
    ``pid_water`` 0.05496, this 0.02774.
    """

    # --- 1. band -----------------------------------------------------------
    # (handled in get_action; both windows pinned to the comfort band)

    # --- 2. fan ------------------------------------------------------------
    #: Normalised static pressure while a handler is heating. The simulator
    #: holds a minimum under any running handler, so -1.0 asks for that
    #: minimum rather than for no air.
    HEATING_PRESSURE = -1.0

    # --- 3. per-zone PI ----------------------------------------------------
    ZONE_KP = 0.1
    ZONE_KI = 0.005
    ZONE_INTEGRAL_LIMIT = 200.0
    PRESTART_HOURS = 1.0
    #: Shadows PIDProportionalReheatPolicy's 2.0 K. Throttling the valve
    #: protects a mild climate from overshoot and starves a cold one.
    REHEAT_FULL_K = 0.5

    # --- 4. cooling hold ---------------------------------------------------
    HOLD_STEPS = 288  # 24 hours

    @property
    def name(self) -> str:
        return "final"

    def __init__(self, building_meta_data, action_spec, *args, **kwargs) -> None:
        super().__init__(building_meta_data, action_spec, *args, **kwargs)

        self._comfort_band = tuple(self.comfort_temp_window)
        self.comfort_now_idx = list(self.field_names).index("comfort_mode_now")

        self._zone_integral: dict[str, float] = {}
        self._pi_active = False

        n = len(self.ahu_controllers)
        self._hold_left = [0] * n
        self._held_action: list[tuple[float, float, float] | None] = [None] * n

    # ----------------------------------------------------------------- gate

    def _in_pi_window(self, obs: np.ndarray) -> bool:
        """From an hour before the comfort window until it closes."""
        if float(obs[self.comfort_now_idx]) > 0.5:
            return True
        hour, day = self._get_hour_and_day(obs)
        if day >= 5:
            return False
        return (self.morning_start_hour - self.PRESTART_HOURS
                <= hour < self.evening_start_hour)

    # -------------------------------------------------------------- zone law

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        if not self._pi_active:
            return super()._command_zones(continuous, zones, obs, low, high)

        centre = 0.5 * (low + high)
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue

            error = self._denormalize_zone_temp(obs[obs_idx]) - centre
            integral = self._zone_integral.get(vav_id, 0.0)

            # Conditional integration: only accumulate while the output is off
            # its limits or the error would unwind it, so a zone the plant
            # cannot reach does not wind up for hours and then take as long to
            # let go once it finally arrives.
            candidate = integral + error
            unclipped = self.ZONE_KP * error + self.ZONE_KI * candidate
            if abs(unclipped) < 1.0 or (unclipped > 0) != (error > 0):
                integral = float(np.clip(
                    candidate, -self.ZONE_INTEGRAL_LIMIT, self.ZONE_INTEGRAL_LIMIT))
            self._zone_integral[vav_id] = integral

            u = float(np.clip(
                self.ZONE_KP * error + self.ZONE_KI * integral, -1.0, 1.0))

            # Air is needed in both directions -- to cool, and to carry the
            # reheat coil's output -- so the damper follows the magnitude and
            # only the valve follows the sign.
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = np.float32(2.0 * abs(u) - 1.0)
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = np.float32(2.0 * max(0.0, -u) - 1.0)

    # ---------------------------------------------------------------- action

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        self._pi_active = self._in_pi_window(obs)
        if not self._pi_active:
            self._zone_integral.clear()

        # Both windows, not just the eco one: the inherited `get_action`
        # chooses between them by rebuilding the work-hour test from the
        # cyclic time features, which cannot see holidays, so on a holiday it
        # would take the comfort branch while the schedule is in eco. Writing
        # the same band into both makes the choice irrelevant.
        self.comfort_temp_window = self._comfort_band
        self.eco_temp_window = self._comfort_band

        action = super().get_action(obs)
        continuous = action["continuous_action"]
        discrete = action["discrete_action"]

        # --- fan clamp, heating only ---
        for ctrl in self.ahu_controllers:
            pressure_idx = ctrl.get("pressure_idx")
            if pressure_idx is None:
                continue
            # An idle handler already has -1 and a cleared run command.
            if not discrete[ctrl["run_discrete_idx"]]:
                continue
            if float(continuous[ctrl["temp_idx"]]) > 0.0:
                continuous[pressure_idx] = np.float32(self.HEATING_PRESSURE)

        # --- cooling-run hold ---
        for i, ctrl in enumerate(self.ahu_controllers):
            run_idx = ctrl["run_discrete_idx"]
            temp_idx = ctrl["temp_idx"]
            pressure_idx = ctrl.get("pressure_idx")
            recirc_idx = ctrl.get("recirc_idx")

            if discrete[run_idx]:
                if float(continuous[temp_idx]) < 0.0:
                    self._hold_left[i] = self.HOLD_STEPS
                    self._held_action[i] = (
                        float(continuous[temp_idx]),
                        float(continuous[pressure_idx])
                        if pressure_idx is not None else 0.0,
                        float(continuous[recirc_idx])
                        if recirc_idx is not None else 1.0,
                    )
                else:
                    self._hold_left[i] = 0
                    self._held_action[i] = None
                continue

            if self._hold_left[i] > 0 and self._held_action[i] is not None:
                temp, pressure, recirc = self._held_action[i]
                discrete[run_idx] = 1
                continuous[temp_idx] = np.float32(temp)
                if pressure_idx is not None:
                    continuous[pressure_idx] = np.float32(pressure)
                if recirc_idx is not None:
                    continuous[recirc_idx] = np.float32(recirc)
                self._hold_left[i] -= 1

        return action
