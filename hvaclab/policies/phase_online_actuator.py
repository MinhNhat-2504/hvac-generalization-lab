"""Ablate only when causal online feedback is applied, retaining learning."""
import numpy as np
from hvaclab.policies.online_actuator import OnlineActuatorPolicy
from hvaclab.policies.heating_damper_curve import HeatingDamperCurvePolicy

PHASES=('prestart','occupied','scheduled_empty','offhours')


class PhaseOnlineActuatorPolicy(OnlineActuatorPolicy):
    def __init__(self,*args,feedback_phase='all',**kwargs):
        if feedback_phase not in ('all','prestart','occupied'):raise ValueError('unknown feedback phase')
        self.feedback_phase=feedback_phase
        super().__init__(*args,**kwargs)
        self.applied_by_phase={p:0 for p in PHASES}

    def reset(self):
        super().reset();self.applied_by_phase={p:0 for p in PHASES}

    def decision_phase(self,obs):
        if not self._in_pi_window(obs):return 'offhours'
        if float(obs[self.comfort_now_idx])<=.5:return 'prestart'
        return 'occupied' if float(obs[self.occupancy_idx])>0 else 'scheduled_empty'

    def _command_zones(self,continuous,zones,obs,low,high):
        phase=self.decision_phase(obs)
        allowed=self.feedback_phase=='all' or phase==self.feedback_phase
        if allowed:
            if self._pi_active:
                self.applied_by_phase[phase]+=sum(self._correction[self._positions[v]]!=0 for v,_,_ in zones)
            return super()._command_zones(continuous,zones,obs,low,high)
        # Bypass feedback only. The estimator still observes and learns from
        # the actual commands issued on this trajectory in every phase.
        return HeatingDamperCurvePolicy._command_zones(self,continuous,zones,obs,low,high)

    def online_report(self):
        report=super().online_report()
        report['proposed_correction_zone_steps']=report.pop('correction_zone_steps')
        report['applied_by_phase']={p:int(n) for p,n in self.applied_by_phase.items()}
        report['actual_correction_zone_steps']=sum(report['applied_by_phase'].values())
        return report
