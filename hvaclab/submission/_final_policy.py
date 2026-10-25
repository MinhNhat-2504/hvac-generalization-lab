"""The controller this entry submits, with no dependency on this repository.

Concatenated after the starter kit's ``pid.py`` and ``pid_plus.py`` by
``hvaclab.submission.build``, so ``PIDWaterResetPolicy`` is already defined
above it in the assembled file. The import below exists only so this file is
readable and type-checkable on its own; the build strips it.

WHY THE DEADLINE MACHINERY IS NOT HERE

``hvaclab.policies.deadline_band`` computes a band that widens when nobody is
being scored and narrows back before the window opens. The measured best
setback depth was zero, which collapses that whole computation to "always use
the comfort band" -- so carrying it into the submission would be shipping a
branch that never fires. The equivalence is asserted in ``build --verify``,
which runs this class against the registry variant and requires the returns to
match exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.submission._shared import PIDWaterResetPolicy

#: The registry key this class must reproduce exactly. ``build --verify``
#: rolls out both and compares.
EQUIVALENT_TO = "x_heatonly"


class FinalPolicy(PIDWaterResetPolicy):
    """``pid_water``, with the comfort band held around the clock, the reheat
    valve saturating sooner, and the fan clamped while heating.

    Three changes, none of them learned, and none of them fitted to a building.

    1. **The band never sets back.** ``pid_water`` widens to an eco band
       outside comfort hours. Holding the comfort band instead measured better
       on every set we tried, and the reason is not an energy argument: a
       building held in band starts the morning with no deviation to recover,
       and the recovery costs more than the setback saves. The shipped
       ``fixed_setpoint`` already hints at this by beating ``pid_water`` in
       cold climates.

    2. **The reheat valve saturates at 0.5 K** instead of 2.0 K. Throttling it
       protects a mild climate from driving a near-miss zone through its band,
       and starves a cold one, where that valve is the only heat source in the
       building and the air reaching it is already 20 K below the zone.

    3. **Static pressure goes to its floor whenever a handler is heating.**
       This is the change that carries the result. The handler cannot heat, and
       below about -6.8 C ambient its supply-air action is inert as well --
       with recirculation capped at 0.70, return air near 293 K and ambient at
       231 K, the mixed air is 274 K, already below the 285 K floor the
       setpoint can command. So raising static pressure pushes more 274 K air
       past the coil, and three tenths of that air is ambient. The fan work is
       spent, and then gas is spent undoing the air the fan moved.

       In cooling the air is the delivery path rather than the load, so the
       same clamp would be a trade instead of a saving, and at 35:1 comfort to
       energy it is a bad one. Measured on held-out buildings, clamping the
       cooling side cost comfort (0.0322 -> 0.0332) for less than it saved. So
       cooling is left exactly as ``pid`` wrote it.

    Per-kWh prices in metric units, which is where "35:1" and "the most
    expensive fuel" come from: fan/pump electricity 0.0064, gas heat 0.0034,
    chiller cooling 0.0018 per kWh thermal at COP 3.5.
    """

    #: Shortfall in K at which the reheat valve is fully open. Shadows
    #: ``PIDProportionalReheatPolicy.REHEAT_FULL_K``, whose default is 2.0.
    REHEAT_FULL_K = 0.5

    #: Normalised static pressure commanded while a handler is heating. -1.0 is
    #: the floor; the simulator holds a minimum static pressure under any
    #: running handler, so this asks for that minimum rather than for no air.
    HEATING_PRESSURE = -1.0

    @property
    def name(self) -> str:
        return "final"

    def __init__(self, building_meta_data, action_spec, *args, **kwargs) -> None:
        super().__init__(building_meta_data, action_spec, *args, **kwargs)
        # The band that is actually scored. Kept because `get_action` overwrites
        # the inherited window attributes on every step and needs the original.
        self._comfort_band = tuple(self.comfort_temp_window)

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        # Both windows are pinned to the comfort band, not just the eco one.
        # The inherited `get_action` chooses between them by rebuilding the
        # work-hour test from the cyclic time features, which cannot see
        # holidays -- so on a holiday it would take the comfort branch while
        # the schedule is in eco. Writing the same band into both makes the
        # choice irrelevant.
        self.comfort_temp_window = self._comfort_band
        self.eco_temp_window = self._comfort_band

        action = super().get_action(obs)
        continuous = action["continuous_action"]
        discrete = action["discrete_action"]

        for ctrl in self.ahu_controllers:
            pressure_idx = ctrl.get("pressure_idx")
            if pressure_idx is None:
                continue
            # An idle handler has already been given -1 and a cleared run
            # command by the inherited law. Leave it: rewriting would risk
            # putting a pressure under a handler that is off.
            if not discrete[ctrl["run_discrete_idx"]]:
                continue
            # The supply-air temperature action is the control effort the
            # inherited law just wrote, so its sign is the direction without
            # recomputing the loop. Positive is heating.
            if float(continuous[ctrl["temp_idx"]]) > 0.0:
                continuous[pressure_idx] = np.float32(self.HEATING_PRESSURE)

        return action
