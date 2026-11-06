"""Isolate late-day allocation while retaining the two-hour morning recovery."""
import numpy as np
from hvaclab.policies.daytime_cooling_lift import DaytimeCoolingLiftPolicy


class CutoffCoolingLiftPolicy(DaytimeCoolingLiftPolicy):
    def __init__(self,*args,cutoff_hour=14.,**kwargs):
        if not np.isfinite(cutoff_hour) or cutoff_hour not in (14.,16.):
            raise ValueError('cutoff_hour must be14 or16')
        self.cutoff_hour=float(cutoff_hour)
        super().__init__(*args,**kwargs)
        if self.morning_start_hour+self.recovery_hours>=self.cutoff_hour:
            raise ValueError('cutoff precedes recovery end')

    def _allocation_window(self,obs):
        hour,_=self._get_hour_and_day(obs)
        return super()._allocation_window(obs) and hour<self.cutoff_hour
