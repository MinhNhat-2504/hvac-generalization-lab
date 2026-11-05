"""Observe cooling-lift actions without feeding physical state to the policy."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import sys
from hvaclab.coverage import write_json


def main():
    from smart_buildings import evaluate as ev
    from hvaclab import offhours_hold_worker as base
    from hvaclab import morning_recovery_worker as trace
    from hvaclab.policies.cooling_lift import CoolingLiftPolicy
    old_register,old_init,old_load=base.register,ev.initialize_policy,ev.load_hybrid_environment
    state={};audit=dict(checked_handler_steps=0,checked_zone_steps=0,max_relative_actual_flow_error=0.,
                       max_relative_estimated_cooling_error=0.,supply_guard_violations=0,max_supply_error_K=0.)
    def register():
        old_register()
        from hvaclab.registry import _OURS,_VARIANTS
        _OURS['cooling_lift']=('hvaclab.policies.cooling_lift','CoolingLiftPolicy')
        for key,flow,lift in (('phase_ctrl',False,False),('flow_reset',True,False),('flow_lift',True,True)):
            _VARIANTS[key]=('cooling_lift',{**_VARIANTS['damper12_mid'][1],
                'offhours_hold_steps':None,'flow_equivalent':flow,'cooling_lift':lift,
                'supply_reduction_k':2.,'pressure_floor_pa':200.,'damper_floor':.1,'target_opening':.98})
    def initialize(cls,meta,spec):
        policy=old_init(cls,meta,spec)
        if isinstance(policy,CoolingLiftPolicy):
            state['policy']=policy;action=policy.get_action
            def capture(obs):
                command=action(obs);state['pending']=dict(policy.last_allocations);state['supplies']={};return command
            policy.get_action=capture
        return policy
    def load(*args,**kw):
        env=old_load(*args,**kw);boxes={v.device_id():v for v in env.building.simulator.hvac.vavs.values()}
        for v,box in boxes.items():
            compute=box.compute_reheat_energy_rate
            def tap(supply,water,_compute=compute,_v=v):
                state.setdefault('supplies',{}).setdefault(_v,float(supply));return _compute(supply,water)
            box.compute_reheat_energy_rate=tap
        reward=env.get_reward_info_and_response
        def capture():
            info,response=reward();pending=state.pop('pending',None)
            if pending:
                p=state['policy'];controllers={c['ahu_id']:c for c in p.ahu_controllers}
                for ahu,allocation in pending.items():
                    audit['checked_handler_steps']+=1;c=controllers[ahu]
                    for j,(v,_,_) in enumerate(c['zones']):
                        box=boxes[v]
                        if abs(box._max_air_flow_static_pressure-p.flow_design_pressure_pa)>1e-8 or abs(box._min_damper_setting-p.damper_floor)>1e-8:
                            raise ValueError('cooling allocation physics assumptions do not match')
                        expected=allocation['requested_flows'][j];actual=float(box.flow_rate_demand)
                        error=abs(actual/expected-1);audit['checked_zone_steps']+=1
                        audit['max_relative_actual_flow_error']=max(audit['max_relative_actual_flow_error'],error)
                        temperature=allocation['observed_temperature'][j]
                        before=allocation['old_flows'][j]*(temperature-allocation['old_supply'])
                        after=actual*(temperature-allocation['new_supply'])
                        audit['max_relative_estimated_cooling_error']=max(audit['max_relative_estimated_cooling_error'],abs(after/before-1))
                        sat_error=abs(state['supplies'][v]-allocation['new_supply'])
                        audit['max_supply_error_K']=max(audit['max_supply_error_K'],sat_error)
                        audit['supply_guard_violations']+=int(sat_error>.01)
            return info,response
        env.get_reward_info_and_response=capture;return env
    base.register,ev.initialize_policy,ev.load_hybrid_environment=register,initialize,load
    try:trace.main()
    finally:base.register,ev.initialize_policy,ev.load_hybrid_environment=old_register,old_init,old_load
    job=json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text());path=Path(job['result_path'])
    record=json.loads(path.read_text());record['lift_audit']=audit;write_json(path,record)
    print('COOLING_LIFT_PHYSICAL_AUDIT',audit,flush=True)


if __name__=='__main__':main()
