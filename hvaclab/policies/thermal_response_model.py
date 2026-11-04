"""Regularized per-zone thermal response conditioned on commanded AHU state.

Fits observable delayed temperature increments only. No simulator state,
scenario identity or reward is needed by this predictor.
"""
import numpy as np

STATES=('off_steady','off_entry','on_steady','on_entry')


def fan_state(on,previous_on):
    return ('on' if on else 'off')+('_entry' if bool(on)!=bool(previous_on) else '_steady')


def features(temperature,slope,occupied,centre=295.5):
    return np.array([1.,(temperature-centre)/3.,np.clip(slope,-2.,2.),float(occupied>0)],dtype=float)


class StateThermalPredictor:
    def __init__(self,centre=295.5,ridge=10.,minimum_zone_samples=8):
        if not np.isfinite(ridge) or ridge<=0:raise ValueError('ridge must be positive')
        self.centre=float(centre);self.ridge=float(ridge)
        self.minimum_zone_samples=minimum_zone_samples
        self.pooled={};self.zones={};self.counts={}

    def fit(self,examples):
        # Example: zone index, state, observed temperature, observed slope,
        # building occupancy, next available sensor increment.
        groups={}
        for zone,state,temp,slope,occupied,delta in examples:
            if state not in STATES:raise ValueError('invalid fan state')
            x=features(temp,slope,occupied,self.centre)
            if not np.all(np.isfinite(x)) or not np.isfinite(delta):continue
            groups.setdefault((int(zone),state),[]).append((x,float(delta)))
        self.pooled.clear();self.zones.clear();self.counts.clear()
        def solve(rows,prior,strength):
            x=np.array([r[0] for r in rows]);y=np.array([r[1] for r in rows])
            return np.linalg.solve(x.T@x+strength*np.eye(4),x.T@y+strength*prior)
        for state in STATES:
            rows=[r for (zone,s),values in groups.items() if s==state for r in values]
            if rows:self.pooled[state]=solve(rows,np.zeros(4),1.)
        for key,rows in groups.items():
            self.counts[key]=len(rows)
            if len(rows)>=self.minimum_zone_samples:
                self.zones[key]=solve(rows,self.pooled[key[1]],self.ridge)
        return self

    def predict_delta(self,zone,state,temperature,slope,occupied):
        if state not in STATES:raise ValueError('invalid fan state')
        coefficients=self.zones.get((int(zone),state),self.pooled.get(state))
        if coefficients is None:return None
        x=features(temperature,slope,occupied,self.centre)
        if not np.all(np.isfinite(x)):return None
        return float(x@coefficients)

    def parameters(self):
        return dict(centre=self.centre,ridge=self.ridge,minimum_zone_samples=self.minimum_zone_samples,
                    pooled={k:v.tolist() for k,v in self.pooled.items()},
                    zones={str(z)+':'+s:v.tolist() for (z,s),v in self.zones.items()},
                    counts={str(z)+':'+s:n for (z,s),n in self.counts.items()})
