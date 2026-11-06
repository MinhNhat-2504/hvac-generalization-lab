"""Target bounded cooling reserve at hot zones using only public/controller state.

Kept outside policies/ so adding this experiment does not change the input
enumeration of the ongoing, frozen cutoff study.
"""
import numpy as np
from hvaclab.policies.guarded_cooling_lift import GuardedCoolingLiftPolicy


class ZoneReservePolicy(GuardedCoolingLiftPolicy):
    def __init__(self,*args,reserve_fraction=.1,**kwargs):
        if not np.isfinite(reserve_fraction) or not 0<=reserve_fraction<=.2:
            raise ValueError('reserve_fraction must be in[0,.2]')
        self.reserve_fraction=float(reserve_fraction)
        super().__init__(*args,**kwargs)
        self.reserve_zone_steps=0;self.reserve_vetoes=0

    def reset(self):
        super().reset();self.reserve_zone_steps=0;self.reserve_vetoes=0

    def get_action(self,obs):
        action=super().get_action(obs)
        if self.reserve_fraction==0 or not self._pi_active:return action
        cont=action['continuous_action']
        for h,c in enumerate(self.ahu_controllers):
            allocation=self.last_allocations.get(c['ahu_id'])
            if allocation is None:continue
            planning=np.array(allocation['planning_temperature'])
            need=np.clip((planning-(self._comfort_band[1]-.5))/.5,0.,1.)
            need*=np.array([self._zone_u.get(v,0.)>0 for v,_,_ in c['zones']])
            multiplier=1+self.reserve_fraction*need
            if not np.any(multiplier>1):continue
            desired=np.array(allocation['requested_flows'])*multiplier
            if desired.sum()>.95*allocation['public_capacity']:
                self.reserve_vetoes+=1;continue
            _,_,meters=self._lift_bindings[h]
            maximum=np.array([obs[i] for i in meters],dtype=float)
            old_pressure=self.native_pressure(h,float(self.last_before_lift_action['continuous_action'][c['pressure_idx']]))
            pressure=max(self.pressure_floor_pa,float(np.max((desired/maximum/self.target_opening)**2*self.flow_design_pressure_pa)))
            if pressure>old_pressure+1e-6:
                self.reserve_vetoes+=1;continue
            norm=self._pressure_norms[h]
            pn=np.float32(norm['min_normalized_value']+(pressure-norm['min_native_value'])/(norm['max_native_value']-norm['min_native_value'])*(norm['max_normalized_value']-norm['min_normalized_value']))
            pressure=self.native_pressure(h,float(pn))
            openings=desired/(maximum*np.sqrt(pressure/self.flow_design_pressure_pa))
            if not np.all(np.isfinite(openings)) or np.any(openings<self.damper_floor) or np.any(openings>1):
                self.reserve_vetoes+=1;continue
            dampers=np.asarray(2*openings-1,dtype=np.float32)
            realized=np.maximum(self.damper_floor,(dampers.astype(float)+1)/2)*maximum*np.sqrt(pressure/self.flow_design_pressure_pa)
            if np.max(np.abs(realized/desired-1))>1e-6:
                self.reserve_vetoes+=1;continue
            indices=[a['supply_air_damper_percentage_command'] for _,_,a in c['zones']]
            cont[c['pressure_idx']]=pn;cont[indices]=dampers
            allocation.update(requested_flows=desired.tolist(),realized_flows=realized.tolist(),
                              target_cooling_multiplier=multiplier.tolist())
            self.reserve_zone_steps+=int(np.sum(multiplier>1))
        return action

    def hold_report(self):
        return dict(super().hold_report(),reserve_zone_steps=self.reserve_zone_steps,reserve_vetoes=self.reserve_vetoes)
