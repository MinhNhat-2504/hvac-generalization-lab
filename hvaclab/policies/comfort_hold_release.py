"""Prepared candidate: release stale cooling persistence under sustained cold.

Only the inherited hold memory is cleared. Actual parent requests, including
new cooling requests, and heat/water coordination continue to own the actions.
"""
import numpy as np

from hvaclab.policies.reheat_curve import ReheatCurvePolicy


class ComfortHoldReleasePolicy(ReheatCurvePolicy):
    def __init__(self,*args,release_enabled=True,cold_deficit_k=.5,
                 confirmation_steps=3,**kwargs):
        if not np.isfinite(cold_deficit_k) or cold_deficit_k<=0:
            raise ValueError('cold deficit must be finite and positive')
        if isinstance(confirmation_steps,bool) or int(confirmation_steps)!=confirmation_steps or confirmation_steps<1:
            raise ValueError('confirmation steps must be a positive integer')
        super().__init__(*args,**kwargs)
        self.release_enabled=bool(release_enabled)
        self.cold_deficit_k=float(cold_deficit_k)
        self.confirmation_steps=int(confirmation_steps)
        self._cold_streak=[0]*len(self.ahu_controllers)
        self.holds_released=0

    def get_action(self,obs):
        low,high=self._comfort_band
        centre=.5*(low+high)
        active=self.release_enabled and self._in_pi_window(obs)
        for i,ctrl in enumerate(self.ahu_controllers):
            temps=[self._denormalize_zone_temp(obs[j]) for j in ctrl['zone_obs_indices']]
            # Requiring every zone at/below the centre deliberately protects
            # cooling reserve even before a hot zone reaches the upper edge.
            cold=(active and bool(temps) and all(np.isfinite(t) for t in temps)
                  and max(temps)<=centre and min(temps)<=low-self.cold_deficit_k)
            self._cold_streak[i]=self._cold_streak[i]+1 if cold else 0
            if (self._cold_streak[i]>=self.confirmation_steps
                    and self._hold_left[i]>0 and self._held_action[i] is not None):
                self._hold_left[i]=0
                self._held_action[i]=None
                self.holds_released+=1
        return super().get_action(obs)
