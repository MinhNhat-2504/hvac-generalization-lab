"""Limit new cooling-hold refreshes only outside the protected PI window."""
import numpy as np
from hvaclab.policies.heating_damper_curve import HeatingDamperCurvePolicy


class OffhoursHoldPolicy(HeatingDamperCurvePolicy):
    def __init__(self,*args,offhours_hold_steps=None,**kwargs):
        if offhours_hold_steps is not None and (isinstance(offhours_hold_steps,bool)
            or not isinstance(offhours_hold_steps,int) or offhours_hold_steps<1):
            raise ValueError('offhours cap must be a positive integer or None')
        self.offhours_hold_steps=offhours_hold_steps
        super().__init__(*args,**kwargs)
        self.limited_decision_steps=0;self.eligible_decision_steps=0
        self.last_effective_hold_steps=self.hold_steps;self.last_limited=False

    def reset(self):
        super().reset()
        self.limited_decision_steps=0;self.eligible_decision_steps=0;self.last_limited=False
        if hasattr(self,'_hold_left'):
            self._hold_left=[0]*len(self.ahu_controllers);self._held_action=[None]*len(self.ahu_controllers)
        if hasattr(self,'_zone_integral'):self._zone_integral.clear();self._zone_u.clear()
        if hasattr(self,'_water_integral'):self._water_integral=0.

    def decision_phase(self,obs):
        if not self._in_pi_window(obs):return 'offhours'
        if float(obs[self.comfort_now_idx])<=.5:return 'prestart'
        return 'occupied' if float(obs[self.occupancy_idx])>0 else 'scheduled_empty'

    def get_action(self,obs):
        occupancy=float(obs[self.occupancy_idx]);normal=self.hold_steps
        eligible=not self._in_pi_window(obs) and np.isfinite(occupancy) and occupancy<=0.
        self.eligible_decision_steps+=int(eligible)
        effective=min(normal,self.offhours_hold_steps) if eligible and self.offhours_hold_steps is not None else normal
        self.last_effective_hold_steps=effective;self.last_limited=effective<normal
        self.limited_decision_steps+=int(self.last_limited)
        # Only duration used for a NEW cooling request changes. Existing
        # inherited hold memory is not trimmed or restored at phase boundaries.
        self.hold_steps=effective
        try:return super().get_action(obs)
        finally:self.hold_steps=normal

    def hold_report(self):
        return dict(limited_decision_steps=self.limited_decision_steps,eligible_decision_steps=self.eligible_decision_steps,
                    normal_hold_steps=self.hold_steps,offhours_hold_steps=self.offhours_hold_steps,
                    holds_started=self.holds_started,holds_applied=self.holds_applied)
