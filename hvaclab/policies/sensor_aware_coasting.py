"""Bounded short-test controller with sensor-age guard and owned AHU recovery.

Only public observations/actions are used. Linear age correction remains a
heuristic; the last active thermal slope need not predict a fan-off interval.
"""
import numpy as np
from hvaclab.policies.thermal_coasting import ThermalCoastingPolicy
from hvaclab.policies.heating_damper_curve import HeatingDamperCurvePolicy


class SensorAwareCoastingPolicy(ThermalCoastingPolicy):
    def __init__(self,*args,sensor_age_steps=1,recovery_steps=6,
                 maximum_age_correction_k=.5,**kwargs):
        if sensor_age_steps not in (0,1):raise ValueError('sensor age must be 0 or 1')
        if isinstance(recovery_steps,bool) or not isinstance(recovery_steps,int) or recovery_steps<1:
            raise ValueError('recovery steps must be a positive integer')
        if not np.isfinite(maximum_age_correction_k) or maximum_age_correction_k<=0:
            raise ValueError('invalid age correction bound')
        super().__init__(*args,**kwargs)
        self.sensor_age_steps=sensor_age_steps
        self.recovery_steps=recovery_steps
        self.maximum_age_correction_k=maximum_age_correction_k
        n=len(self.ahu_controllers)
        self._recovery_left=[0]*n
        self._resume_commands=[None]*n
        self._resume_water=[{} for _ in range(n)]
        self.guard_releases=0
        self.recovery_run_steps=0
        self.forced_run_steps=0

    def get_action(self,obs):
        if self.coast_mode=='off':return HeatingDamperCurvePolicy.get_action(self,obs)
        integral_before=dict(self._zone_integral)
        ahu_before={i:dict(state) for i,state in self.pid_state.items()}
        water_before=self._water_integral
        action=HeatingDamperCurvePolicy.get_action(self,obs)
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
            safe=allowed and bool(ctrl['zones']) and self._recovery_left[i]<=0
            for vav,index,_ in ctrl['zones']:
                temp=float(self._denormalize_zone_temp(obs[index]))
                now[vav]=temp
                previous=self._previous_zone_temps.get(vav)
                if not np.isfinite(temp) or previous is None or not np.isfinite(previous):
                    safe=False
                    continue
                slope=float(np.clip(temp-previous,-self.maximum_age_correction_k,self.maximum_age_correction_k))
                current=temp+self.sensor_age_steps*slope
                projected=current+self.forecast_steps*slope
                safe=bool(safe and low+self.coast_margin_k<=current<=high-self.coast_margin_k
                          and low+self.coast_margin_k<=projected<=high-self.coast_margin_k)
            run=ctrl['run_discrete_idx']
            was_coasting=self._coasting[i]
            if disc[run] and not was_coasting and self._recovery_left[i]<=0:
                indices=[ctrl['temp_idx'],ctrl.get('pressure_idx'),ctrl.get('recirc_idx')]
                for _,_,zone_indices in ctrl['zones']:indices.extend(zone_indices.values())
                self._resume_commands[i]={j:float(cont[j]) for j in indices if j is not None}
                self._resume_water[i]={j:float(cont[j]) for j in (self.water_temp_idx,self.water_pressure_idx) if j is not None}
            coast=bool(safe and self._resume_commands[i] is not None and
                       (was_coasting or self._on_steps[i]>=self.minimum_on_steps))
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
                self._recovery_left[i]=self.recovery_steps
                self.guard_releases+=1
            if not coast and self._recovery_left[i]>0:
                # Own the actual run command for the bounded recovery period.
                # Replay known running SAT/pressure/recirculation; zone PI still
                # owns its current dampers and valves during its active window.
                resume=self._resume_commands[i]
                if resume is None:raise RuntimeError('coast recovery without prior running command')
                was_off=not bool(disc[run])
                disc[run]=1
                if was_off:
                    for j in (ctrl['temp_idx'],ctrl.get('pressure_idx'),ctrl.get('recirc_idx')):
                        if j is not None:cont[j]=np.float32(resume[j])
                    if any(self._zone_u.get(v,0.)<0 for v,_,_ in ctrl['zones']):
                        cont[ctrl['temp_idx']]=np.float32(1.)
                        if ctrl.get('pressure_idx') is not None:cont[ctrl['pressure_idx']]=np.float32(-1.)
                        if ctrl.get('recirc_idx') is not None:cont[ctrl['recirc_idx']]=np.float32(1.)
                if not self._pi_active:
                    for _,_,indices in ctrl['zones']:
                        for j in indices.values():cont[j]=np.float32(resume[j])
                needs_water=any(float(cont[indices['reheat_valve_percentage_command']])>-1.+1e-6
                                for _,_,indices in ctrl['zones'] if 'reheat_valve_percentage_command' in indices)
                if needs_water and 0<=self.water_discrete_idx<self.n_discrete:
                    disc[self.water_discrete_idx]=1
                    for j,value in self._resume_water[i].items():cont[j]=np.float32(max(float(cont[j]),value))
                self.recovery_run_steps+=1
                self.forced_run_steps+=int(was_off)
                self._recovery_left[i]-=1
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
