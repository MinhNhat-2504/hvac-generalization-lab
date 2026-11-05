"""Bounded smoothing of AHU aggregate error, leaving per-zone PI untouched."""
import numpy as np
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy


class AHUDampedPolicy(FlowEquivalentPolicy):
    def __init__(self,*args,ahu_damping=True,ahu_alpha=.5,ahu_max_correction_k=.15,ahu_urgent_k=1.,**kwargs):
        if not np.isfinite(ahu_alpha) or not 0<ahu_alpha<=1:
            raise ValueError('AHU alpha must be in(0,1]')
        if not np.isfinite(ahu_max_correction_k) or ahu_max_correction_k<0:
            raise ValueError('AHU correction must be finite and nonnegative')
        if not np.isfinite(ahu_urgent_k) or ahu_urgent_k<=0:
            raise ValueError('AHU urgent deviation must be positive')
        self.ahu_damping=bool(ahu_damping);self.ahu_alpha=float(ahu_alpha)
        self.ahu_max_correction_k=float(ahu_max_correction_k);self.ahu_urgent_k=float(ahu_urgent_k)
        super().__init__(*args,**kwargs)
        self._clear_ahu_history()

    def _clear_ahu_history(self):
        self._ahu_filtered={};self.ahu_filtered_steps=0;self.ahu_urgent_bypasses=0
        self.ahu_max_applied_correction=0.

    def reset(self):
        super().reset();self._clear_ahu_history()

    def _ahu_error(self,obs,ctrl,low,high):
        raw=float(super()._ahu_error(obs,ctrl,low,high));key=ctrl['ahu_id']
        previous=self._ahu_filtered.get(key)
        identity=not self.ahu_damping or self.ahu_alpha==1 or self.ahu_max_correction_k==0
        urgent=abs(raw)>=self.ahu_urgent_k
        if identity or not np.isfinite(raw) or previous is None or raw==0 or urgent or raw*previous<=0:
            value=raw
            if urgent and not identity:self.ahu_urgent_bypasses+=1
        else:
            value=float(np.clip(self.ahu_alpha*raw+(1-self.ahu_alpha)*previous,
                                raw-self.ahu_max_correction_k,raw+self.ahu_max_correction_k))
            if value!=raw:
                self.ahu_filtered_steps+=1
                self.ahu_max_applied_correction=max(self.ahu_max_applied_correction,abs(value-raw))
        if np.isfinite(value):self._ahu_filtered[key]=value
        else:self._ahu_filtered.pop(key,None)
        return value

    def hold_report(self):
        return dict(super().hold_report(),ahu_filtered_steps=self.ahu_filtered_steps,
                    ahu_urgent_bypasses=self.ahu_urgent_bypasses,
                    ahu_max_applied_correction=self.ahu_max_applied_correction)
