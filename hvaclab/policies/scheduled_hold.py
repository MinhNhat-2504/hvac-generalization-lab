"""Bound cooling-run persistence outside the PI/preconditioning window.

This changes the duration of a previously requested cooling run only. The
flat temperature band, genuine nighttime heating/cooling calls, daytime PI,
and reheat airflow/water coordination remain the inherited laws.
"""
from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class ScheduledHoldPolicy(CoordinatedHeatPolicy):
    def __init__(self, *args, off_window_hold_steps=12, **kwargs):
        if int(off_window_hold_steps) != off_window_hold_steps or off_window_hold_steps < 0:
            raise ValueError("off-window hold must be a nonnegative number of steps")
        super().__init__(*args, **kwargs)
        self.off_window_hold_steps = int(off_window_hold_steps)

    def get_action(self, obs):
        outside = not self._in_pi_window(obs)
        if outside:
            self._hold_left = [min(n, self.off_window_hold_steps) for n in self._hold_left]
        action = super().get_action(obs)
        if outside:
            # An actual cooling call may have rearmed the regular 288-step
            # hold. Bound that new timer too, without suppressing the call.
            self._hold_left = [min(n, self.off_window_hold_steps) for n in self._hold_left]
        return action
