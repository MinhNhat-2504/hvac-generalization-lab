"""Per-handler thermal coasting with a forecast-band restart guard.

The extrapolation is a short-horizon heuristic from observable temperatures,
not an identified physical model or a comfort guarantee. Parent V10 commands
are used whenever the guard cannot justify coasting.
"""
import numpy as np

from hvaclab.policies.heating_damper_curve import HeatingDamperCurvePolicy


class ThermalCoastingPolicy(HeatingDamperCurvePolicy):
    def __init__(self,*args,coast_mode='off',coast_margin_k=.75,
                 forecast_steps=3,minimum_on_steps=6,**kwargs):
        if coast_mode not in ('off','empty','all'):
            raise ValueError('unknown coasting mode')
        if not np.isfinite(coast_margin_k) or coast_margin_k<=0:
            raise ValueError('margin must be finite and positive')
        for value in (forecast_steps,minimum_on_steps):
            if isinstance(value,bool) or not np.isfinite(value) or int(value)!=value or value<1:
                raise ValueError('step counts must be positive integers')
        super().__init__(*args,**kwargs)
        if 2*coast_margin_k>=self._comfort_band[1]-self._comfort_band[0]:
            raise ValueError('coasting margin leaves no interior comfort band')
        self.coast_mode=coast_mode
        self.coast_margin_k=float(coast_margin_k)
        self.forecast_steps=int(forecast_steps)
        self.minimum_on_steps=int(minimum_on_steps)
        self._previous_zone_temps={}
        self._coasting=[False]*len(self.ahu_controllers)
        self._on_steps=[0]*len(self.ahu_controllers)
        self.coast_handler_steps=0
        self.coast_entries=0
        self.coast_occupied_handler_steps=0
        self.guard_restarts=0

    def get_action(self,obs):
        if self.coast_mode=='off':return super().get_action(obs)
        integral_before=dict(self._zone_integral)
        ahu_before={i:dict(state) for i,state in self.pid_state.items()}
        water_before=self._water_integral
        action=super().get_action(obs)
        cont,disc=action['continuous_action'],action['discrete_action']
        low,high=self._comfort_band
        # Reserve the inherited PI preconditioning window. The occupancy
        # feature is used only as a nonnegative zero/nonzero observation.
        occupied=float(obs[self.occupancy_idx])
        comfort=bool(float(obs[self.comfort_now_idx])>.5)
        allowed=not (self._pi_active and not comfort)
        if self.coast_mode=='empty':
            allowed=allowed and not self._pi_active and np.isfinite(occupied) and occupied<=1e-6
        now={}
        for i,ctrl in enumerate(self.ahu_controllers):
            safe=allowed and bool(ctrl['zones'])
            for vav,index,_ in ctrl['zones']:
                temp=float(self._denormalize_zone_temp(obs[index]))
                now[vav]=temp
                previous=self._previous_zone_temps.get(vav)
                if not np.isfinite(temp) or previous is None or not np.isfinite(previous):
                    safe=False
                    continue
                projected=temp+self.forecast_steps*(temp-previous)
                safe=bool(safe and low+self.coast_margin_k<=temp<=high-self.coast_margin_k
                          and low+self.coast_margin_k<=projected<=high-self.coast_margin_k)
            run=ctrl['run_discrete_idx']
            was_coasting=self._coasting[i]
            coast=bool(safe and (was_coasting or self._on_steps[i]>=self.minimum_on_steps))
            if coast:
                disc[run]=0
                cont[ctrl['temp_idx']]=np.float32(1.)
                for field,value in (('pressure_idx',-1.),('recirc_idx',1.)):
                    index=ctrl.get(field)
                    if index is not None:cont[index]=np.float32(value)
                for vav,_,indices in ctrl['zones']:
                    for key in ('supply_air_damper_percentage_command','reheat_valve_percentage_command'):
                        index=indices.get(key)
                        if index is not None:cont[index]=np.float32(-1.)
                    if self._pi_active:
                        if vav in integral_before:self._zone_integral[vav]=integral_before[vav]
                        else:self._zone_integral.pop(vav,None)
                self.pid_state[i].update(ahu_before[i])
                self._hold_left[i]=0
                self._held_action[i]=None
                self.coast_handler_steps+=1
                self.coast_occupied_handler_steps+=int(np.isfinite(occupied) and occupied>0)
                self.coast_entries+=int(not was_coasting)
            elif was_coasting:
                self.guard_restarts+=1
            self._coasting[i]=coast
            self._on_steps[i]=self._on_steps[i]+1 if disc[run] else 0
        self._previous_zone_temps=now
        if not any(disc[c['run_discrete_idx']] for c in self.ahu_controllers):
            # No air can carry heat: close all water valves above and turn
            # off the pump. Never disable shared water while another AHU runs.
            if 0<=self.water_discrete_idx<self.n_discrete:disc[self.water_discrete_idx]=0
            for index in (self.water_temp_idx,self.water_pressure_idx):
                if index is not None:cont[index]=np.float32(-1.)
            if any(self._coasting):self._water_integral=water_before
        return action
