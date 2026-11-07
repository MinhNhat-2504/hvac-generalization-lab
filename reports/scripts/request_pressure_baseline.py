"""Request-driven pressure-only comparison, not a full ASHRAE Guideline 36 sequence.

Post-processes fixed V10 commands. Damper/SAT/reheat/water commands remain as
constructed by V10, so changing pressure can change airflow and comfort.
Five-minute timing and reset magnitudes are declared experimental choices.
"""
from __future__ import annotations

import numpy as np

from hvaclab.policies.offhours_hold import OffhoursHoldPolicy


class RequestPressureBaseline(OffhoursHoldPolicy):
    """Trim pressure absent downstream requests; respond to high damper demand.

    Each running handler starts at the inherited pressure, bounded to public
    native limits and the declared floor. At later update instants, each zone
    requests one pressure increment while its inherited damper command is at
    least request_on; it keeps requesting until below request_off. There is no
    SAT reset, damper compensation, flow-equivalence or comfort certificate.
    """

    def __init__(
        self, *args, request_reset=True, pressure_floor_pa=200.,
        trim_pa=10., respond_pa=15., max_response_pa=37.,
        request_on=.95, request_off=.85, ignored_requests=0,
        update_steps=1, startup_delay_steps=1, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for value in (pressure_floor_pa, trim_pa, respond_pa, max_response_pa):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('pressure settings must be finite and positive')
        if not (0 <= request_off < request_on <= 1):
            raise ValueError('invalid request hysteresis')
        for name,value,minimum in (('ignored_requests',ignored_requests,0),
                                   ('update_steps',update_steps,1),
                                   ('startup_delay_steps',startup_delay_steps,0)):
            if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
                raise ValueError('invalid '+name)
        self.request_reset=bool(request_reset)
        self.pressure_floor_pa=float(pressure_floor_pa)
        self.trim_pa=float(trim_pa);self.respond_pa=float(respond_pa)
        self.max_response_pa=float(max_response_pa)
        self.request_on=float(request_on);self.request_off=float(request_off)
        self.ignored_requests=ignored_requests;self.update_steps=update_steps
        self.startup_delay_steps=startup_delay_steps
        norms=self.metadata.get('action_normalization_metadata',{})
        self._request_norms=[norms.get(c['ahu_id']+'_supply_air_static_pressure_setpoint',
                            norms.get('supply_air_static_pressure_setpoint')) for c in self.ahu_controllers]
        self._clear_requests()

    def _clear_requests(self):
        n=len(self.ahu_controllers)
        self._request_pressure=[None]*n;self._request_run_steps=[0]*n
        self._zone_requests=[[False]*len(c['zones']) for c in self.ahu_controllers]
        self.trim_updates=0;self.respond_updates=0;self.request_handler_steps=0
        self.unsupported_handler_steps=0;self.last_request_counts=[0]*n
        self.last_parent_action=None

    def reset(self):
        super().reset();self._clear_requests()

    def _forget_handler(self,h):
        self._request_pressure[h]=None;self._request_run_steps[h]=0
        self._zone_requests[h]=[False]*len(self.ahu_controllers[h]['zones'])
        self.last_request_counts[h]=0

    def _binding(self,h):
        norm=self._request_norms[h];c=self.ahu_controllers[h]
        keys=('min_native_value','max_native_value','min_normalized_value','max_normalized_value')
        if not norm or not all(k in norm and np.isfinite(norm[k]) for k in keys):return None
        low,high=norm['min_native_value'],norm['max_native_value']
        nlow,nhigh=norm['min_normalized_value'],norm['max_normalized_value']
        indices=[a.get('supply_air_damper_percentage_command') for _,_,a in c['zones']]
        if (c.get('pressure_idx') is None or not indices or any(i is None for i in indices)
                or high<=low or nhigh<=nlow or high<self.pressure_floor_pa):return None
        return norm,indices,max(low,self.pressure_floor_pa),high

    def get_action(self,obs):
        action=super().get_action(obs)
        self.last_parent_action={k:np.array(v,copy=True) for k,v in action.items()}
        if not self.request_reset:return action
        continuous=action['continuous_action'];discrete=action['discrete_action']
        for h,c in enumerate(self.ahu_controllers):
            if not discrete[c['run_discrete_idx']]:
                self._forget_handler(h);continue
            binding=self._binding(h)
            if binding is None:
                self._forget_handler(h);self.unsupported_handler_steps+=1;continue
            norm,indices,low,high=binding;pidx=c['pressure_idx']
            inherited=float(continuous[pidx]);damper=np.asarray(continuous[indices],dtype=float)
            if not np.isfinite(inherited) or not np.all(np.isfinite(damper)) or np.any(damper < -1) or np.any(damper > 1):
                self._forget_handler(h);self.unsupported_handler_steps+=1;continue
            opening=(damper+1)/2
            requests=[bool(value>=self.request_on or (previous and value>=self.request_off))
                      for value,previous in zip(opening,self._zone_requests[h])]
            self._zone_requests[h]=requests;count=sum(requests);self.last_request_counts[h]=count
            old=self._request_pressure[h];age=self._request_run_steps[h]
            if old is None:
                fraction=(inherited-norm['min_normalized_value'])/(norm['max_normalized_value']-norm['min_normalized_value'])
                pressure=float(np.clip(norm['min_native_value']+fraction*(norm['max_native_value']-norm['min_native_value']),low,high))
            else:
                pressure=old
                if age>=self.startup_delay_steps and (age-self.startup_delay_steps)%self.update_steps==0:
                    if count<=self.ignored_requests:
                        pressure=max(low,old-self.trim_pa);self.trim_updates+=1
                    else:
                        pressure=min(high,old+min(self.max_response_pa,self.respond_pa*(count-self.ignored_requests)))
                        self.respond_updates+=1
            normalized=norm['min_normalized_value']+(pressure-norm['min_native_value'])/(norm['max_native_value']-norm['min_native_value'])*(norm['max_normalized_value']-norm['min_normalized_value'])
            continuous[pidx]=np.float32(np.clip(normalized,norm['min_normalized_value'],norm['max_normalized_value']))
            self._request_pressure[h]=pressure;self._request_run_steps[h]=age+1
            self.request_handler_steps+=1
        return action

    def hold_report(self):
        return dict(super().hold_report(),request_pressure_trim_updates=self.trim_updates,
                    request_pressure_respond_updates=self.respond_updates,
                    request_pressure_handler_steps=self.request_handler_steps,
                    request_pressure_unsupported_steps=self.unsupported_handler_steps)
