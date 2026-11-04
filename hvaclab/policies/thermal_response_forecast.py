"""Public-input two-step temperature forecasts with explicit unknown intervals.

This advisory predictor does not command HVAC. Empirical radii are diagnostic,
not guaranteed safety bounds. A controller must check every served zone.
"""
import numpy as np
from hvaclab.policies.thermal_response_model import fan_state


class ThermalResponseForecaster:
    def __init__(self,model,radii,band=(294.,297.),transition_only=True):
        self.model=model;self.radii=dict(radii);self.band=tuple(band)
        self.transition_only=bool(transition_only)
        if len(self.band)!=2 or not self.band[0]<self.band[1]:raise ValueError('invalid comfort band')
        for radius in self.radii.values():
            if not np.isfinite(radius) or radius<0:raise ValueError('invalid residual radius')

    def predict(self,zone,observed,previous_observed,occupied,previous_on,before_previous_on,next_on):
        if not np.all(np.isfinite([observed,previous_observed,occupied])):return None
        slope=float(observed-previous_observed)
        state_now=fan_state(previous_on,before_previous_on)
        state_next=fan_state(next_on,previous_on)
        first=self.model.predict_delta(zone,state_now,observed,slope,occupied)
        if first is None:return None
        current=observed+first
        second=self.model.predict_delta(zone,state_next,current,first,occupied)
        if second is None:return None
        point=current+second
        if self.transition_only and state_next not in ('off_entry','on_entry'):
            point=observed+2*slope
        radius=self.radii.get(state_next)
        interval=None if radius is None else (point-radius,point+radius)
        allowed=bool(not next_on and interval is not None
                     and self.band[0]<=observed<=self.band[1]
                     and self.band[0]<=current<=self.band[1]
                     and self.band[0]<=interval[0] and interval[1]<=self.band[1])
        return dict(estimated_current=float(current),next_temperature=float(point),state=state_next,
                    interval=interval,interval_allows_coast=allowed)
