"""Delay both fan reset and cooling allocation until morning recovery ends."""
import numpy as np
from hvaclab.policies.guarded_cooling_lift import GuardedCoolingLiftPolicy


class DaytimeCoolingLiftPolicy(GuardedCoolingLiftPolicy):
    def __init__(self,*args,recovery_hours=2.,**kwargs):
        if not np.isfinite(recovery_hours) or not 0<=recovery_hours<=4:
            raise ValueError('recovery_hours must be in[0,4]')
        self.recovery_hours=float(recovery_hours)
        super().__init__(*args,**kwargs)

    def _allocation_window(self,obs):
        hour,_=self._get_hour_and_day(obs)
        return bool(float(obs[self.comfort_now_idx])>.5 and
                    self.morning_start_hour+self.recovery_hours<=hour<self.evening_start_hour)

    def get_action(self,obs):
        flow,lift=self.flow_equivalent,self.cooling_lift
        allowed=self._allocation_window(obs)
        self.flow_equivalent=flow and allowed
        self.cooling_lift=lift and allowed
        try:return super().get_action(obs)
        finally:self.flow_equivalent,self.cooling_lift=flow,lift
