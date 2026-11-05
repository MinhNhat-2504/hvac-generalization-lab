"""Lower fan pressure while preserving every zone's commanded airflow."""
import numpy as np
from hvaclab.policies.offhours_hold import OffhoursHoldPolicy


class FlowEquivalentPolicy(OffhoursHoldPolicy):
    def __init__(self,*args,flow_equivalent=True,pressure_floor_pa=200.,damper_floor=.1,target_opening=.98,**kwargs):
        if not np.isfinite(pressure_floor_pa) or pressure_floor_pa<=0:
            raise ValueError('running pressure floor must be positive')
        if not 0<damper_floor<=target_opening<=1:
            raise ValueError('invalid damper bounds')
        self.flow_equivalent=bool(flow_equivalent)
        self.pressure_floor_pa=float(pressure_floor_pa)
        self.damper_floor=float(damper_floor);self.target_opening=float(target_opening)
        super().__init__(*args,**kwargs)
        norms=self.metadata.get('action_normalization_metadata',{})
        self._pressure_norms=[norms.get(c['ahu_id']+'_supply_air_static_pressure_setpoint',
                              norms.get('supply_air_static_pressure_setpoint')) for c in self.ahu_controllers]
        self._clear_flow_diagnostics()

    def _clear_flow_diagnostics(self):
        self.rebalanced_handler_steps=0;self.max_relative_flow_error=0.;self.last_parent_action=None

    def reset(self):
        super().reset();self._clear_flow_diagnostics()

    def native_pressure(self,h,value):
        norm=self._pressure_norms[h]
        f=(value-norm['min_normalized_value'])/(norm['max_normalized_value']-norm['min_normalized_value'])
        return max(self.pressure_floor_pa,norm['min_native_value']+f*(norm['max_native_value']-norm['min_native_value']))

    def get_action(self,obs):
        action=super().get_action(obs)
        self.last_parent_action={k:np.array(v,copy=True) for k,v in action.items()}
        if not self.flow_equivalent:return action
        cont,disc=action['continuous_action'],action['discrete_action']
        for h,c in enumerate(self.ahu_controllers):
            norm=self._pressure_norms[h];pidx=c.get('pressure_idx')
            if not norm or pidx is None or not c['zones'] or not disc[c['run_discrete_idx']]:continue
            keys=('min_native_value','max_native_value','min_normalized_value','max_normalized_value')
            if not all(k in norm and np.isfinite(norm[k]) for k in keys):continue
            span=norm['max_native_value']-norm['min_native_value'];nspan=norm['max_normalized_value']-norm['min_normalized_value']
            if span<=0 or nspan<=0 or norm['min_native_value']>self.pressure_floor_pa:continue
            indices=[a.get('supply_air_damper_percentage_command') for _,_,a in c['zones']]
            if any(i is None for i in indices):continue
            opening=np.maximum(self.damper_floor,(cont[indices].astype(float)+1)/2)
            if not np.all(np.isfinite(opening)) or not np.isfinite(cont[pidx]):continue
            old=self.native_pressure(h,float(cont[pidx]))
            proposed=max(self.pressure_floor_pa,old*(float(opening.max())/self.target_opening)**2)
            if proposed>=old*(1-1e-7):continue
            normalized=norm['min_normalized_value']+(proposed-norm['min_native_value'])/span*nspan
            normalized=np.float32(np.clip(normalized,norm['min_normalized_value'],norm['max_normalized_value']))
            new=self.native_pressure(h,float(normalized))
            if new>=old:continue
            replacement=np.asarray(2*opening*np.sqrt(old/new)-1,dtype=np.float32)
            if np.any(replacement>1) or np.any(replacement<-1):continue
            realized=np.maximum(self.damper_floor,(replacement.astype(float)+1)/2)*np.sqrt(new)
            desired=opening*np.sqrt(old)
            error=float(np.max(np.abs(realized/desired-1)))
            if error>1e-6:continue
            cont[pidx]=normalized;cont[indices]=replacement
            self.rebalanced_handler_steps+=1
            self.max_relative_flow_error=max(self.max_relative_flow_error,error)
        return action

    def hold_report(self):
        return dict(super().hold_report(),rebalanced_handler_steps=self.rebalanced_handler_steps,
                    max_relative_flow_error=self.max_relative_flow_error)
