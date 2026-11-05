"""Coordinate supply-temperature reduction and room airflow using public inputs."""
import numpy as np
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy


class CoolingLiftPolicy(FlowEquivalentPolicy):
    def __init__(self,*args,cooling_lift=False,supply_reduction_k=2.,flow_design_pressure_pa=2000.,**kwargs):
        if not np.isfinite(supply_reduction_k) or not 0<supply_reduction_k<=2:
            raise ValueError('supply reduction must be in(0,2]K')
        if not np.isfinite(flow_design_pressure_pa) or flow_design_pressure_pa<=0:raise ValueError('invalid flow design pressure')
        self.cooling_lift=bool(cooling_lift);self.supply_reduction_k=float(supply_reduction_k)
        self.flow_design_pressure_pa=float(flow_design_pressure_pa)
        super().__init__(*args,**kwargs)
        norms=self.metadata.get('action_normalization_metadata',{});names=list(self.field_names)
        self._lift_bindings=[]
        for c in self.ahu_controllers:
            prefix='ahs_'+c['ahu_id']+'_'
            required=[prefix+'supply_air_flowrate_sensor',prefix+'supply_fan_speed_percentage_command']
            qnames=[v+'_supply_air_flowrate_setpoint' for v,_,_ in c['zones']]
            norm=norms.get(c['ahu_id']+'_supply_air_temperature_setpoint',norms.get('supply_air_temperature_setpoint'))
            supported=norm is not None and c.get('oat_idx') is not None and all(n in names for n in required+qnames)
            self._lift_bindings.append((norm,[names.index(n) for n in required],[names.index(n) for n in qnames]) if supported else None)
        self._clear_lift()

    def _clear_lift(self):
        self.lift_handler_steps=0;self.lift_capacity_vetoes=0;self.lift_floor_vetoes=0
        self.last_allocations={};self.last_before_lift_action=None

    def reset(self):super().reset();self._clear_lift()

    @staticmethod
    def native(norm,value):
        span=norm['max_normalized_value']-norm['min_normalized_value']
        if span<=0:raise ValueError('invalid normalization')
        return norm['min_native_value']+(value-norm['min_normalized_value'])/span*(norm['max_native_value']-norm['min_native_value'])

    def get_action(self,obs):
        action=super().get_action(obs);self.last_allocations={}
        self.last_before_lift_action={k:np.array(v,copy=True) for k,v in action.items()}
        if not self.cooling_lift:return action
        cont,disc=action['continuous_action'],action['discrete_action']
        for h,c in enumerate(self.ahu_controllers):
            binding=self._lift_bindings[h];pidx=c.get('pressure_idx')
            if binding is None or pidx is None or self._pressure_norms[h] is None or not c['zones'] or not disc[c['run_discrete_idx']]:continue
            norm,meters,qindices=binding
            if not all(k in norm and np.isfinite(norm[k]) for k in ('min_native_value','max_native_value','min_normalized_value','max_normalized_value')):continue
            if norm['max_native_value']<=norm['min_native_value'] or norm['max_normalized_value']<=norm['min_normalized_value']:continue
            old_supply=self.native(norm,float(cont[c['temp_idx']]))
            temperatures=np.array([self._denormalize_zone_temp(obs[i]) for _,i,_ in c['zones']],dtype=float)
            outside=float(obs[c['oat_idx']]);max_flows=np.array([obs[i] for i in qindices],dtype=float)
            if not np.all(np.isfinite(np.r_[temperatures,max_flows,outside])) or np.any(max_flows<=0):continue
            if np.min(temperatures)<max(self._comfort_band[0]+.5,old_supply+3.) or outside<old_supply+2.:continue
            indices=[a.get('supply_air_damper_percentage_command') for _,_,a in c['zones']]
            valves=[a.get('reheat_valve_percentage_command') for _,_,a in c['zones']]
            if any(i is None for i in indices+valves) or any(cont[i]>-1+1e-7 for i in valves):continue
            observed_flow,observed_speed=(float(obs[i]) for i in meters)
            if not np.isfinite(observed_flow+observed_speed) or observed_flow<=.01 or not .001<observed_speed<=1.0001:continue
            capacity=observed_flow/observed_speed
            old_pressure=self.native_pressure(h,float(cont[pidx]))
            openings=np.maximum(self.damper_floor,(cont[indices].astype(float)+1)/2)
            old_flows=openings*max_flows*np.sqrt(old_pressure/self.flow_design_pressure_pa)
            if not np.all(np.isfinite(old_flows)) or old_flows.sum()>.95*capacity:
                self.lift_capacity_vetoes+=1;continue
            desired_supply=max(norm['min_native_value'],old_supply-self.supply_reduction_k)
            if desired_supply>=old_supply-.01:continue
            new_sat=np.float32(norm['min_normalized_value']+(desired_supply-norm['min_native_value'])/(norm['max_native_value']-norm['min_native_value'])*(norm['max_normalized_value']-norm['min_normalized_value']))
            new_supply=self.native(norm,float(new_sat))
            requested_flows=old_flows*(temperatures-old_supply)/(temperatures-new_supply)
            requested_pressure=max(self.pressure_floor_pa,float(np.max((requested_flows/max_flows/self.target_opening)**2*self.flow_design_pressure_pa)))
            if requested_pressure>old_pressure+1e-6:continue
            pn=self._pressure_norms[h]
            new_p=np.float32(pn['min_normalized_value']+(requested_pressure-pn['min_native_value'])/(pn['max_native_value']-pn['min_native_value'])*(pn['max_normalized_value']-pn['min_normalized_value']))
            new_pressure=self.native_pressure(h,float(new_p))
            replacement=requested_flows/(max_flows*np.sqrt(new_pressure/self.flow_design_pressure_pa))
            if np.any(replacement<self.damper_floor) or np.any(replacement>1):
                self.lift_floor_vetoes+=1;continue
            new_d=np.asarray(2*replacement-1,dtype=np.float32)
            cont[c['temp_idx']]=new_sat;cont[pidx]=new_p;cont[indices]=new_d
            realized=np.maximum(self.damper_floor,(new_d.astype(float)+1)/2)*max_flows*np.sqrt(new_pressure/self.flow_design_pressure_pa)
            self.last_allocations[c['ahu_id']]=dict(old_supply=old_supply,new_supply=new_supply,
                old_flows=old_flows.tolist(),requested_flows=requested_flows.tolist(),realized_flows=realized.tolist(),
                observed_temperature=temperatures.tolist(),public_capacity=capacity)
            self.lift_handler_steps+=1
        return action

    def hold_report(self):
        return dict(super().hold_report(),lift_handler_steps=self.lift_handler_steps,
                    lift_capacity_vetoes=self.lift_capacity_vetoes,lift_floor_vetoes=self.lift_floor_vetoes)
