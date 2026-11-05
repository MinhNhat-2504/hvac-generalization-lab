"""Closed-loop flow-equivalent fan reset with offline physical-flow auditing."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import sys
import numpy as np
from hvaclab.coverage import write_json


def main(register_extra=None):
    from smart_buildings import evaluate as ev
    from hvaclab import offhours_hold_worker as base
    from hvaclab import morning_recovery_worker as trace
    from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
    old_register,old_init,old_load=base.register,ev.initialize_policy,ev.load_hybrid_environment
    state={};audit=dict(checked_zone_steps=0,max_relative_actual_flow_error=0.,
                       parent_pressure_fan_Wh=0.,actual_fan_Wh=0.,changed_handler_steps=0)

    def register():
        old_register()
        from hvaclab.registry import _OURS,_VARIANTS
        _OURS['flow_equivalent']=('hvaclab.policies.flow_equivalent','FlowEquivalentPolicy')
        for key,enabled in (('phase_ctrl',False),('flow_reset',True)):
            _VARIANTS[key]=('flow_equivalent',{**_VARIANTS['damper12_mid'][1],
               'offhours_hold_steps':None,'flow_equivalent':enabled,'pressure_floor_pa':200.,'damper_floor':.1,'target_opening':.98})
        if register_extra is not None:register_extra()

    def initialize(cls,meta,spec):
        policy=old_init(cls,meta,spec)
        if isinstance(policy,FlowEquivalentPolicy):
            state['policy']=policy;action=policy.get_action
            def capture(obs):
                command=action(obs)
                state['pending']=(policy.last_parent_action,command)
                return command
            policy.get_action=capture
        return policy

    def load(*args,**kwargs):
        env=old_load(*args,**kwargs);reward=env.get_reward_info_and_response
        def capture():
            info,response=reward();pending=state.pop('pending',None)
            if pending is not None:
                parent,command=pending;p=state['policy'];hvac=env.building.simulator.hvac
                ahus={v.device_id():v for v in getattr(hvac.air_handler,'ahus',[hvac.air_handler])}
                vavs={v.device_id():v for v in hvac.vavs.values()}
                np.testing.assert_array_equal(parent['discrete_action'],command['discrete_action'])
                allowed={i for c in p.ahu_controllers for i in [c.get('pressure_idx')]+[a.get('supply_air_damper_percentage_command') for _,_,a in c['zones']] if i is not None}
                keep=[i for i in range(p.n_continuous) if i not in allowed]
                np.testing.assert_array_equal(parent['continuous_action'][keep],command['continuous_action'][keep])
                dt=(info.end_timestamp.seconds-info.start_timestamp.seconds+(info.end_timestamp.nanos-info.start_timestamp.nanos)/1e9)/3600.
                for h,c in enumerate(p.ahu_controllers):
                    if not command['discrete_action'][c['run_discrete_idx']]:continue
                    ahu=ahus[c['ahu_id']]
                    if abs(ahu._min_supply_air_static_pressure-p.pressure_floor_pa)>1e-8:raise ValueError('assumed AHU floor differs from actual model')
                    if p._pressure_norms[h] is None:raise ValueError('missing public pressure normalization')
                    old_p=p.native_pressure(h,float(parent['continuous_action'][c['pressure_idx']]))
                    new_p=p.native_pressure(h,float(command['continuous_action'][c['pressure_idx']]))
                    if new_p>old_p+1e-3:raise ValueError('fan pressure increased')
                    audit['changed_handler_steps']+=int(new_p<old_p-1e-3)
                    for v,_,a in c['zones']:
                        box=vavs[v]
                        if abs(box._min_damper_setting-p.damper_floor)>1e-8:raise ValueError('assumed VAV floor differs from actual model')
                        opening=max(box._min_damper_setting,(float(parent['continuous_action'][a['supply_air_damper_percentage_command']])+1)/2)
                        expected=max(opening*box._max_air_flow_rate*np.sqrt(old_p/box._max_air_flow_static_pressure),.00001)
                        error=abs(box.flow_rate_demand/expected-1)
                        if error>2e-6:raise ValueError('actual room airflow not preserved '+str(error))
                        audit['checked_zone_steps']+=1
                        audit['max_relative_actual_flow_error']=max(audit['max_relative_actual_flow_error'],float(error))
                    parent_power=(ahu.air_flow_rate+ahu.ambient_flow_rate)*old_p/ahu.fan_efficiency
                    actual_power=ahu.compute_intake_fan_energy_rate()+ahu.compute_exhaust_fan_energy_rate()
                    audit['parent_pressure_fan_Wh']+=parent_power*dt;audit['actual_fan_Wh']+=actual_power*dt
            return info,response
        env.get_reward_info_and_response=capture;return env

    base.register,ev.initialize_policy,ev.load_hybrid_environment=register,initialize,load
    try:trace.main()
    finally:base.register,ev.initialize_policy,ev.load_hybrid_environment=old_register,old_init,old_load
    job=json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text())
    output=Path(job['result_path']);record=json.loads(output.read_text());record['flow_audit']=audit
    write_json(output,record)
    print('ACTUAL_FLOW_EQUIVALENCE_VERIFIED',audit,flush=True)


if __name__=='__main__':main()
