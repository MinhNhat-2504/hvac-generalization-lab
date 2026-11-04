"""Causal online actuator estimator with bounded zone-PI-only feedback.

Learns an increment only after its next public sensor arrives. No offline
coefficients, scenario identifiers, simulator state or rewards are used.
"""
import numpy as np
from hvaclab.policies.heating_damper_curve import HeatingDamperCurvePolicy
from hvaclab.policies.actuator_thermal_model import actuator_features,FEATURE_NAMES
from hvaclab.policies.thermal_response_model import fan_state,STATES


class OnlineActuatorEstimator:
    def __init__(self,zones,minimum_samples=24,decay=.95):
        self.zones=zones;self.minimum_samples=minimum_samples;self.decay=decay
        self.reset()

    def reset(self):
        n=self.zones;d=len(FEATURE_NAMES)
        self.weights=np.zeros((n,8,d))
        self.covariance=np.broadcast_to(np.eye(d)/10.,(n,8,d,d)).copy()
        self.counts=np.zeros((n,8),dtype=int)
        self.error=np.zeros((8,3));self.error_steps=np.zeros(8,dtype=int)
        self.pending=None
        self.mae_sum=np.zeros((2,4));self.mae_count=np.zeros(2,dtype=int)

    def observe(self,temperatures):
        if self.pending is None:return
        pending=self.pending;target=temperatures-pending['temperature']
        for group in np.unique(pending['groups']):
            indices=np.flatnonzero((pending['groups']==group)&pending['ready']&np.isfinite(target))
            if len(indices):
                residual=np.abs(target[indices,None]-pending['experts'][indices])
                mean=residual.mean(axis=0)
                self.error[group]=mean if self.error_steps[group]==0 else self.decay*self.error[group]+(1-self.decay)*mean
                self.error_steps[group]+=1
        for z,group in enumerate(pending['groups']):
            if not np.isfinite(target[z]):continue
            x=pending['x'][z];cov=self.covariance[z,group];projection=cov@x
            gain=projection/(1.+x@projection)
            self.weights[z,group]+=gain*(target[z]-x@self.weights[z,group])
            updated=cov-np.outer(gain,projection)
            self.covariance[z,group]=(updated+updated.T)/2
            self.counts[z,group]+=1
            if pending['ready'][z]:
                occupancy=int(group>=4)
                predictions=np.r_[pending['experts'][z],pending['selected'][z]]
                self.mae_sum[occupancy]+=np.abs(target[z]-predictions)
                self.mae_count[occupancy]+=1
        self.pending=None

    def predict(self,temperatures,slopes,features,states,occupied):
        groups=np.array([STATES.index(state)+4*int(occupied>0) for state in states])
        selected=np.zeros(self.zones);experts=np.zeros((self.zones,3));ready=np.zeros(self.zones,dtype=bool)
        chosen=np.zeros(self.zones,dtype=int)
        for z,group in enumerate(groups):
            experts[z]=[0.,slopes[z],self.weights[z,group]@features[z]]
            ready[z]=self.counts[z,group]>=self.minimum_samples
            if ready[z] and self.error_steps[group]>=self.minimum_samples:
                baseline=int(np.argmin(self.error[group,:2]))
                best=2 if self.error[group,2]<.95*self.error[group,baseline] else baseline
                # Refuse untrustworthy corrections rather than extrapolate them.
                if self.error[group,best]<=.25:chosen[z]=best
            selected[z]=experts[z,chosen[z]]
        self.pending=dict(temperature=temperatures.copy(),x=features.copy(),groups=groups,
                          experts=experts,ready=ready,selected=selected.copy())
        return selected,chosen

    def report(self):
        return dict(learned_samples=int(self.counts.sum()),group_frames=self.error_steps.tolist(),
            prequential_mae={name:(self.mae_sum[i]/self.mae_count[i]).tolist() if self.mae_count[i] else None for i,name in enumerate(('empty','occupied'))},
            scored_zone_steps=self.mae_count.tolist(),expert_order=['aged_sensor','linear','actuator','selected'])


class OnlineActuatorPolicy(HeatingDamperCurvePolicy):
    def __init__(self,*args,online_blend=0.,online_limit_k=.5,online_warmup=288,**kwargs):
        if not 0<=online_blend<=1 or not np.isfinite(online_limit_k) or online_limit_k<=0 or online_warmup<0:
            raise ValueError('invalid online correction settings')
        super().__init__(*args,**kwargs)
        self.online_blend=float(online_blend);self.online_limit_k=float(online_limit_k);self.online_warmup=int(online_warmup)
        self._online_zones=[(v,index,indices,ctrl) for ctrl in self.ahu_controllers for v,index,indices in ctrl['zones']]
        self._positions={v:i for i,(v,_,_,_) in enumerate(self._online_zones)}
        self._bindings=[];self._supported=True
        for ctrl in self.ahu_controllers:
            prefix='ahs_'+ctrl['ahu_id']+'_'
            names=[prefix+suffix for suffix in ('supply_air_temperature_sensor','supply_air_static_pressure_sensor','recirculation_percentage_command','outside_air_temperature_sensor')]
            if not all(name in self.field_names for name in names):self._supported=False;self._bindings.append(None)
            else:self._bindings.append(tuple(self.field_names.index(name) for name in names))
        self._water_norm=self.metadata.get('action_normalization_metadata',{}).get('supply_water_temperature_setpoint')
        self._supported=self._supported and self._water_norm is not None
        self.estimator=OnlineActuatorEstimator(len(self._online_zones))
        self._clear_history()

    def _clear_history(self):
        self._last_action=None;self._older_action=None;self._previous_temps=None;self._online_steps=0
        n=len(getattr(self,'_online_zones',[]));self._correction=np.zeros(n)
        self.correction_zone_steps=0;self.supported_feature_steps=0;self.chosen_experts=np.zeros(3,dtype=int)

    def reset(self):
        super().reset()
        if hasattr(self,'estimator'):
            self.estimator.reset();self._clear_history()
            self._zone_integral.clear();self._zone_u.clear()
            self._hold_left=[0]*len(self.ahu_controllers);self._held_action=[None]*len(self.ahu_controllers)
            self._water_integral=0.

    def _context(self,obs,indices,ctrl):
        cont=self._last_action['continuous_action'];disc=self._last_action['discrete_action']
        h=self.ahu_controllers.index(ctrl);sat,p,recirc,oat=self._bindings[h]
        if not 100<float(obs[sat])<400 or not 100<float(obs[oat])<400:raise ValueError('AHU native temperature unavailable')
        norm=self._water_norm;water=float(cont[self.water_temp_idx])
        fraction=(water-norm['min_normalized_value'])/(norm['max_normalized_value']-norm['min_normalized_value'])
        return dict(fan=int(disc[ctrl['run_discrete_idx']]),pump=int(disc[self.water_discrete_idx]),
            sat_command=float(cont[ctrl['temp_idx']]),pressure_command=float(cont[ctrl['pressure_idx']]),recirc_command=float(cont[ctrl['recirc_idx']]),
            damper=(float(cont[indices['supply_air_damper_percentage_command']])+1)/2,
            valve=(float(cont[indices['reheat_valve_percentage_command']])+1)/2,
            water_temperature_command=water,water_pressure_command=float(cont[self.water_pressure_idx]),
            water_temperature_native=norm['min_native_value']+fraction*(norm['max_native_value']-norm['min_native_value']),
            sat_proxy_k=float(obs[sat]),outside_k=float(obs[oat]),pressure_observation=float(obs[p]),recirc_observation=float(obs[recirc]))

    def get_action(self,obs):
        temperature=np.array([self._denormalize_zone_temp(obs[index]) for _,index,_,_ in self._online_zones],dtype=float)
        self._correction.fill(0.)
        if self._supported and np.all(np.isfinite(temperature)):
            self.estimator.observe(temperature)
            if self._previous_temps is not None and self._last_action is not None and self._older_action is not None:
                slopes=temperature-self._previous_temps;states=[];rows=[]
                try:
                    for z,(_,_,indices,ctrl) in enumerate(self._online_zones):
                        states.append(fan_state(self._last_action['discrete_action'][ctrl['run_discrete_idx']],self._older_action['discrete_action'][ctrl['run_discrete_idx']]))
                        rows.append(actuator_features(temperature[z],slopes[z],float(obs[self.occupancy_idx]),self._context(obs,indices,ctrl),sum(self._comfort_band)/2))
                    features=np.asarray(rows)
                    if not np.all(np.isfinite(features)):raise ValueError('non-finite feature frame')
                    selected,chosen=self.estimator.predict(temperature,slopes,features,states,float(obs[self.occupancy_idx]))
                    self.chosen_experts+=np.bincount(chosen,minlength=3);self.supported_feature_steps+=1
                    if self._online_steps>=self.online_warmup:
                        self._correction=np.clip(self.online_blend*selected,-self.online_limit_k,self.online_limit_k)
                except (ValueError,KeyError,TypeError):self.estimator.pending=None
        else:self.estimator.pending=None
        action=super().get_action(obs)
        if self._pi_active:self.correction_zone_steps+=int(np.count_nonzero(self._correction))
        self._older_action=self._last_action
        self._last_action={key:np.array(value,copy=True) for key,value in action.items()}
        self._previous_temps=temperature.copy() if np.all(np.isfinite(temperature)) else None
        self._online_steps+=1
        return action

    def _command_zones(self,continuous,zones,obs,low,high):
        if not self._pi_active or not np.any(self._correction):return super()._command_zones(continuous,zones,obs,low,high)
        corrected=np.array(obs,copy=True)
        for v,index,_ in zones:corrected[index]+=self._correction[self._positions[v]]/self.zone_air_temperature_sensor_std
        return super()._command_zones(continuous,zones,corrected,low,high)

    def online_report(self):
        return dict(**self.estimator.report(),supported_feature_steps=self.supported_feature_steps,
                    correction_zone_steps=self.correction_zone_steps,chosen_experts=self.chosen_experts.tolist())
