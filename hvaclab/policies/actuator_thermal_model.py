"""Public-actuator thermal increment features and regularized per-state model."""
import numpy as np
from hvaclab.policies.thermal_response_model import STATES

FEATURE_NAMES=('bias','temperature','trend','occupied','fan','sat_command','pressure_command',
 'recirc_command','damper','valve','pump','water_temperature_command','water_pressure_command',
 'outdoor_lift','sat_setpoint_lift','pressure_observation','recirc_observation','fan_damper',
 'fan_damper_pressure','fan_damper_sat_lift','fan_water_damper_valve','valve_water_temperature',
 'mixed_cooling_proxy','reheat_lift_proxy','valve_pressure_proxy','fan_trend')


def actuator_features(temperature,slope,occupied,context,centre=295.5):
    c=context;fan=float(c['fan']);pump=float(c['pump'])
    damper=float(c['damper']);valve=float(c['valve'])
    sat_lift=(c['sat_proxy_k']-temperature)/10.
    oat_lift=(c['outside_k']-temperature)/20.
    recirc=float(c['recirc_observation'])
    mixed=recirc*temperature+(1.-recirc)*c['outside_k']
    supply_proxy=min(mixed,c['sat_proxy_k'])
    flow=fan*max(.1,damper)*np.sqrt(max(0.,c['pressure_observation'])/1000.)
    water_fraction=max(0.,(c['water_pressure_command']+1.)/2.)
    heat=flow*pump*valve*(c['water_temperature_native']-temperature)/50.
    return np.array([1.,(temperature-centre)/3.,np.clip(slope,-2.,2.),float(occupied>0),fan,
        c['sat_command'],c['pressure_command'],c['recirc_command'],damper,valve,pump,
        c['water_temperature_command'],c['water_pressure_command'],oat_lift,sat_lift,
        c['pressure_observation']/1000.,recirc,fan*damper,fan*damper*(c['pressure_command']+1.)/2.,
        fan*damper*sat_lift,fan*pump*damper*valve,fan*pump*damper*valve*c['water_temperature_command'],
        flow*(supply_proxy-temperature)/10.,heat,fan*pump*valve*np.sqrt(water_fraction),
        fan*np.clip(slope,-2.,2.)],dtype=float)


class ActuatorThermalPredictor:
    def __init__(self,ridge=10.,minimum_zone_samples=8):
        if not np.isfinite(ridge) or ridge<=0:raise ValueError('invalid regularization')
        self.ridge=float(ridge);self.minimum_zone_samples=minimum_zone_samples
        self.pooled={};self.zones={}

    def fit(self,examples):
        groups={}
        for zone,state,x,y in examples:
            x=np.asarray(x,dtype=float)
            if state not in STATES or x.shape!=(len(FEATURE_NAMES),):raise ValueError('invalid example')
            if np.all(np.isfinite(x)) and np.isfinite(y):groups.setdefault((zone,state),[]).append((x,float(y)))
        self.pooled.clear();self.zones.clear();size=len(FEATURE_NAMES)
        def solve(rows,prior,penalty):
            x=np.array([r[0] for r in rows]);y=np.array([r[1] for r in rows])
            return np.linalg.solve(x.T@x+penalty*np.eye(size),x.T@y+penalty*prior)
        for state in STATES:
            rows=[r for (z,s),values in groups.items() if s==state for r in values]
            if rows:self.pooled[state]=solve(rows,np.zeros(size),1.)
        for key,rows in groups.items():
            if len(rows)>=self.minimum_zone_samples:self.zones[key]=solve(rows,self.pooled[key[1]],self.ridge)
        return self

    def predict_delta(self,zone,state,x):
        if state not in STATES:raise ValueError('invalid state')
        coefficient=self.zones.get((zone,state),self.pooled.get(state))
        x=np.asarray(x,dtype=float)
        if coefficient is None or x.shape!=(len(FEATURE_NAMES),) or not np.all(np.isfinite(x)):return None
        return float(x@coefficient)

    def parameters(self):
        return dict(feature_names=FEATURE_NAMES,ridge=self.ridge,
            pooled={s:x.tolist() for s,x in self.pooled.items()},
            zones={str(z)+':'+s:x.tolist() for (z,s),x in self.zones.items()})
