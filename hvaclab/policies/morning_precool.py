"""Public-temperature-gated early PI recovery on the short offhours hold."""
import numpy as np
from hvaclab.policies.offhours_hold import OffhoursHoldPolicy


class MorningPrecoolPolicy(OffhoursHoldPolicy):
    def __init__(self,*args,recovery_extra_hours=1.,recovery_warm_margin=.5,recovery_cold_margin=.5,**kwargs):
        if not np.isfinite(recovery_extra_hours) or not 0<=recovery_extra_hours<=2:
            raise ValueError('extra recovery must be between zero and two hours')
        for value in (recovery_warm_margin,recovery_cold_margin):
            if not np.isfinite(value) or value<0:
                raise ValueError('recovery temperature margins must be finite and nonnegative')
        self.recovery_extra_hours=float(recovery_extra_hours)
        self.recovery_warm_margin=float(recovery_warm_margin)
        self.recovery_cold_margin=float(recovery_cold_margin)
        self.recovery_steps=0
        super().__init__(*args,**kwargs)

    def _early_recovery(self,obs):
        if self.recovery_extra_hours==0 or super()._in_pi_window(obs):
            return False
        hour,day=self._get_hour_and_day(obs)
        normal_start=self.morning_start_hour-self.prestart_hours
        if day>=5 or not normal_start-self.recovery_extra_hours<=hour<normal_start:
            return False
        occupancy=float(obs[self.occupancy_idx])
        if not np.isfinite(occupancy) or occupancy!=0:
            return False
        values=np.array([self._denormalize_zone_temp(obs[i]) for c in self.ahu_controllers for _,i,_ in c['zones']])
        if not len(values) or not np.all(np.isfinite(values)):
            return False
        low,high=self._comfort_band
        return bool(values.mean()>=.5*(low+high)+self.recovery_warm_margin
                    and values.min()>=low+self.recovery_cold_margin)

    def _in_pi_window(self,obs):
        return super()._in_pi_window(obs) or self._early_recovery(obs)

    def get_action(self,obs):
        self.recovery_steps+=int(self._early_recovery(obs))
        return super().get_action(obs)

    def reset(self):
        super().reset();self.recovery_steps=0

    def hold_report(self):
        return dict(super().hold_report(),recovery_steps=self.recovery_steps,
                    recovery_extra_hours=self.recovery_extra_hours)
