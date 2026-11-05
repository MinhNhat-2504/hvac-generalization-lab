"""Bounded causal signed temperature projection and an optional comfort guard."""
import numpy as np
from hvaclab.policies.cooling_lift import CoolingLiftPolicy


class GuardedCoolingLiftPolicy(CoolingLiftPolicy):
    def __init__(self,*args,project_lift=False,comfort_guard=False,projection_limit_k=.5,
                 projection_steps=2.,guard_margin_k=.25,**kwargs):
        if not np.isfinite(projection_limit_k) or not 0<=projection_limit_k<=.5:raise ValueError('projection must be in[0,.5]K')
        if not np.isfinite(projection_steps) or not 0<=projection_steps<=2:raise ValueError('projection horizon must be in[0,2] steps')
        if not np.isfinite(guard_margin_k) or not 0<=guard_margin_k<=.5:raise ValueError('guard margin must be in[0,.5]K')
        self.project_lift=bool(project_lift);self.comfort_guard=bool(comfort_guard)
        self.projection_limit_k=float(projection_limit_k);self.projection_steps=float(projection_steps)
        self.guard_margin_k=float(guard_margin_k)
        super().__init__(*args,**kwargs);self._clear_projection()

    def _clear_projection(self):
        self._previous_lift_temperatures={};self._lift_projection={}
        self.projected_zone_steps=0;self.max_projection_k=0.;self.comfort_guard_steps=0

    def reset(self):super().reset();self._clear_projection()

    def _planning_temperatures(self,obs,ctrl,temperatures):
        if not self.project_lift:return temperatures
        return temperatures+np.array([self._lift_projection.get(v,0.) for v,_,_ in ctrl['zones']])

    def get_action(self,obs):
        current={v:float(self._denormalize_zone_temp(obs[i])) for c in self.ahu_controllers for v,i,_ in c['zones']}
        self._lift_projection={}
        for v,t in current.items():
            previous=self._previous_lift_temperatures.get(v)
            if (self.project_lift or self.comfort_guard) and previous is not None and np.isfinite(t) and np.isfinite(previous):
                delta=float(np.clip(self.projection_steps*(t-previous),-self.projection_limit_k,self.projection_limit_k))
                self._lift_projection[v]=delta
                self.max_projection_k=max(self.max_projection_k,abs(delta))
        action=super().get_action(obs)
        self.projected_zone_steps+=sum(abs(delta)>1e-8 for delta in self._lift_projection.values()) if self.project_lift else 0
        protected=self._in_pi_window(obs) or float(obs[self.occupancy_idx])>0
        if self.comfort_guard and protected:
            low,high=self._comfort_band
            for c in self.ahu_controllers:
                risk=False
                for v,_,_ in c['zones']:
                    delta=self._lift_projection.get(v,0.);temperature=current[v]
                    risk=risk or (delta<0 and temperature+delta<low+self.guard_margin_k) or (delta>0 and temperature+delta>high-self.guard_margin_k)
                if not risk:continue
                # Restore the same-step raw V10 command, not a cached command
                # from another trajectory. Parent PI/hold state is retained.
                indices=[c.get(k) for k in ('temp_idx','pressure_idx','recirc_idx')]
                indices += [i for _,_,a in c['zones'] for i in a.values()]
                indices=[i for i in indices if i is not None]
                action['continuous_action'][indices]=self.last_parent_action['continuous_action'][indices]
                run=c['run_discrete_idx'];action['discrete_action'][run]=self.last_parent_action['discrete_action'][run]
                if c['ahu_id'] in self.last_allocations:
                    del self.last_allocations[c['ahu_id']];self.lift_handler_steps-=1
                self.comfort_guard_steps+=1
        self._previous_lift_temperatures={v:t for v,t in current.items() if np.isfinite(t)}
        return action

    def hold_report(self):
        return dict(super().hold_report(),projected_zone_steps=self.projected_zone_steps,
                    max_projection_k=self.max_projection_k,comfort_guard_steps=self.comfort_guard_steps)
