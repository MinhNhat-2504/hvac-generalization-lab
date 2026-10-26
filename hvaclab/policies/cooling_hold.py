"""Stop restarting air handlers fifty to a hundred times a day.

THE BEHAVIOUR

The shipped loop idles an air handler the instant its zones are back inside
the band::

    if error == 0.0:
        ...
        discrete[ctrl["run_discrete_idx"]] = 0
        continue

and starts it again at the next excursion. Since the zone loop oscillates
across the band (see ``zone_pi``), those excursions arrive every few steps, and
the public "Probe ZT Best" notebook counted the result: **every air handler
switching on or off 50 to 107 times a day**.

Each restart is paid for twice. The fan spins up against a cold duct, and the
zones it serves drift while it is off, so the restart is bigger than the
excursion that triggered it. Nothing in the reward rewards an idle handler
directly -- only the energy it does not draw -- so an idle second that costs a
larger burst later is a loss.

WHY ONLY THE COOLING RUNS ARE HELD

Holding every run helps hot climates and *hurts* Fairbanks, where it stretches
heating bursts that were better off short: the air handler cannot heat, so a
handler held on in the cold is moving 274 K air past the coils for no reason
while the hot-water loop does the actual work. Holding only the runs that were
started for cooling keeps the hot-climate gain and drops the cold-climate loss.

The direction is read from the supply-air temperature action at the moment the
handler starts, which is the control effort the loop just wrote: negative is
cooling.

Credit: the diagnosis, the asymmetry and the 288-step hold length are from the
public notebook "Smart Buildings: per-zone PI control and AHU cooling hold"
(Probe ZT Best) on the competition's Code tab. The implementation is our own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvaclab.policies.zone_pi import ZonePIPolicy


class CoolingHoldPolicy(ZonePIPolicy):
    """``zone_pi``, plus a minimum run length for handlers started to cool.

    Args:
        hold_steps: How many five-minute steps a cooling run is held at its
            last commands before the loop is allowed to stop it. 288 is a day.
            0 disables the hold, which makes it the ablation control.
    """

    def __init__(
        self,
        building_meta_data,
        action_spec,
        hold_steps: int = 288,
        **kwargs,
    ) -> None:
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.hold_steps = int(hold_steps)

        n = len(self.ahu_controllers)
        #: steps remaining on each handler's hold, 0 when free to stop
        self._hold_left = [0] * n
        #: the commands a held handler repeats
        self._held_action: list[tuple[float, float, float] | None] = [None] * n
        self.holds_started = 0
        self.holds_applied = 0

    @property
    def name(self) -> str:
        return f"cooling_hold_{self.hold_steps}"

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        action = super().get_action(obs)
        if self.hold_steps <= 0:
            return action

        continuous = action["continuous_action"]
        discrete = action["discrete_action"]

        for i, ctrl in enumerate(self.ahu_controllers):
            run_idx = ctrl["run_discrete_idx"]
            temp_idx = ctrl["temp_idx"]
            pressure_idx = ctrl.get("pressure_idx")
            recirc_idx = ctrl.get("recirc_idx")

            running = bool(discrete[run_idx])
            effort = float(continuous[temp_idx])

            if running:
                # A run the loop is asking for. Arm the hold only for cooling:
                # a handler held on in the cold moves 274 K air past the coils
                # while the hot-water loop does the actual work.
                if effort < 0.0:
                    if self._hold_left[i] <= 0:
                        self.holds_started += 1
                    self._hold_left[i] = self.hold_steps
                    self._held_action[i] = (
                        effort,
                        float(continuous[pressure_idx])
                        if pressure_idx is not None else 0.0,
                        float(continuous[recirc_idx])
                        if recirc_idx is not None else 1.0,
                    )
                else:
                    # Heating: let the loop own this handler again.
                    self._hold_left[i] = 0
                    self._held_action[i] = None
                continue

            # The loop wants to stop. Hold it at its last cooling commands
            # until the hold expires.
            if self._hold_left[i] > 0 and self._held_action[i] is not None:
                temp, pressure, recirc = self._held_action[i]
                discrete[run_idx] = 1
                continuous[temp_idx] = np.float32(temp)
                if pressure_idx is not None:
                    continuous[pressure_idx] = np.float32(pressure)
                if recirc_idx is not None:
                    continuous[recirc_idx] = np.float32(recirc)
                self._hold_left[i] -= 1
                self.holds_applied += 1

        return action
