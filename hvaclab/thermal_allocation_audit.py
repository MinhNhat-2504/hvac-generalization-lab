"""Read-only feasibility audit for coordinated thermal lift and room airflow."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from hvaclab.coverage import digest,write_json


def cooling_candidate(temperatures,flows,max_flows,design_pressures,supply,target,
                      pressure_floor=200.,minimum_damper=.1,target_opening=.98):
    t,q,m,p=map(lambda x:np.asarray(x,dtype=float),(temperatures,flows,max_flows,design_pressures))
    if not (len(t) and t.shape==q.shape==m.shape==p.shape):return None
    if not np.all(np.isfinite(np.r_[t,q,m,p,supply,target])) or np.any(q<=0) or np.any(m<=0) or np.any(p<=0):return None
    if target>=supply-.01 or np.any(t<=supply+.5):return None
    proposed=q*(t-supply)/(t-target)
    requested_pressure=max(pressure_floor,float(np.max((proposed/m/target_opening)**2*p)))
    opening=proposed/(m*np.sqrt(requested_pressure/p))
    if np.any(opening<minimum_damper-1e-9) or np.any(opening>1+1e-9):return None
    residual=np.max(np.abs(proposed*(t-target)-q*(t-supply)))
    return dict(flows=proposed.tolist(),pressure=requested_pressure,openings=opening.tolist(),heat_residual=float(residual))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);args=parser.parse_args()
    job_path=Path(args.job);job=json.loads(job_path.read_text());original_argv=sys.argv
    # The observer reuses the verified unmodified V10 worker and does not alter
    # observations, actions, simulation methods' arguments or reward calls.
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.simulator import constants
    from hvaclab import flow_equivalent_worker as worker
    original_load=ev.load_hybrid_environment;state={};rows=[]
    rho_cp=constants.AIR_DENSITY*constants.AIR_HEAT_CAPACITY

    def load(*a,**kw):
        env=original_load(*a,**kw);hvac=env.building.simulator.hvac
        ahus={v.device_id():v for v in getattr(hvac.air_handler,'ahus',[hvac.air_handler])}
        boxes={v.device_id():v for v in hvac.vavs.values()}
        for name,device in ahus.items():
            compute=device.compute_cooling_electrical_energy_rate
            def meter_tap(return_temperature,outside_temperature,_compute=compute,_device=device,_name=name):
                power=_compute(return_temperature,outside_temperature)
                state.setdefault('meters',{}).setdefault(_name,dict(
                    fan_W=float(_device.compute_intake_fan_energy_rate()+_device.compute_exhaust_fan_energy_rate()),
                    cooling_W=abs(float(power)),mixed=float(_device.get_mixed_air_temp(return_temperature,outside_temperature))))
                return power
            device.compute_cooling_electrical_energy_rate=meter_tap
        for name,box in boxes.items():
            compute=box.compute_reheat_energy_rate
            def tap(supply,input_water,_box=box,_compute=compute,_name=name):
                heat=_compute(supply,input_water);q=float(_box.flow_rate_demand)
                state.setdefault('boxes',{}).setdefault(_name,dict(
                    ahu=_box.air_handler.device_id(),zone_temperature=float(_box.zone_air_temperature),
                    flow=q,max_flow=float(_box._max_air_flow_rate),design_pressure=float(_box._max_air_flow_static_pressure),
                    supply=float(supply),water=float(input_water),coil_W=float(heat),
                    valve=float(_box.reheat_valve_setting),damper=float(_box.damper_setting),
                    discharge_limit=float(_box._max_discharge_air_temperature),
                    min_damper=float(_box._min_damper_setting),
                    net_heat_W=float(rho_cp*q*(supply-_box.zone_air_temperature)+heat)))
                return heat
            box.compute_reheat_energy_rate=tap
        step=env.step
        def begin(action):
            state['boxes']={};state['meters']={};state['pending']=True
            return step(action)
        env.step=begin;reward=env.get_reward_info_and_response
        def capture():
            info,response=reward()
            if state.pop('pending',False):
                reward_bytes=info.SerializeToString()
                row=dict(step=len(rows),boxes=state['boxes'],cooling={},heating=[],fan_Wh=0.,cooling_Wh=0.,gas_Wh=0.)
                dt=(info.end_timestamp.seconds-info.start_timestamp.seconds+(info.end_timestamp.nanos-info.start_timestamp.nanos)/1e9)/3600.
                total_fan=sum(float(v.blower_electrical_energy_rate) for v in info.air_handler_reward_infos.values())
                total_cooling=sum(abs(float(v.air_conditioning_electrical_energy_rate)) for v in info.air_handler_reward_infos.values())
                for source,total in (('fan_W',total_fan),('cooling_W',total_cooling)):
                    if abs(sum(v[source] for v in state['meters'].values())-total)>max(.001,2e-6*abs(total)):
                        raise ValueError('per-device and system scorer meters do not reconcile '+source)
                row['fan_Wh']=total_fan*dt;row['cooling_Wh']=total_cooling*dt
                for ahu_id,device in ahus.items():
                    meter=state['meters'][ahu_id];fan=meter['fan_W'];ac=meter['cooling_W']
                    served=[v for v in state['boxes'].values() if v['ahu']==ahu_id]
                    if not served or not int(device.run_command):continue
                    supply=served[0]['supply'];target=max(285.,supply-2.)
                    if ac<=0 or abs(supply-device.supply_air_temperature_setpoint)>1e-6 or any(v['valve']>1e-8 or abs(v['supply']-supply)>1e-8 for v in served):continue
                    proposal=cooling_candidate([v['zone_temperature'] for v in served],[v['flow'] for v in served],
                        [v['max_flow'] for v in served],[v['design_pressure'] for v in served],supply,target,
                        device._min_supply_air_static_pressure,max(v['min_damper'] for v in served))
                    if proposal is None:continue
                    old_pressure=device.supply_air_static_pressure_sensor
                    if proposal['pressure']>old_pressure+1e-3:continue
                    old_flow=sum(v['flow'] for v in served);new_flow=sum(proposal['flows'])
                    mixed=meter['mixed']
                    new_ac=rho_cp*new_flow*(mixed-target)/device._cooling_cop
                    new_fan=new_flow*proposal['pressure']*(2-device.recirculation)/device.fan_efficiency
                    row['cooling'][ahu_id]=dict(fan_saving_Wh=(fan-new_fan)*dt,cooling_saving_Wh=(ac-new_ac)*dt,
                        old_supply=supply,new_supply=target,flow_ratio=new_flow/old_flow,
                        pressure=proposal['pressure'],pointwise_heat_residual=proposal['heat_residual'])
                for v in state['boxes'].values():
                    if v['net_heat_W']<=0 or v['flow']<=0 or v['discharge_limit']<=v['zone_temperature']:continue
                    box_floor=v['min_damper']*v['max_flow']*np.sqrt(ahus[v['ahu']]._min_supply_air_static_pressure/v['design_pressure'])
                    minimum=max(box_floor,v['net_heat_W']/(rho_cp*(v['discharge_limit']-v['zone_temperature'])))
                    row['heating'].append(dict(net_heat_W=v['net_heat_W'],
                        ideal_flow_reduction=max(0.,1-minimum/v['flow']),
                        effective_discharge=v['supply']+v['coil_W']/(rho_cp*v['flow'])))
                row['gas_Wh']=sum(float(v.natural_gas_heating_energy_rate)*dt for v in info.boiler_reward_infos.values())
                if info.SerializeToString()!=reward_bytes:raise ValueError('observer mutated scorer info')
                rows.append(row)
            return info,response
        env.get_reward_info_and_response=capture;return env
    ev.load_hybrid_environment=load
    try:
        sys.argv=[original_argv[0],'--job',str(job_path)];worker.main()
    finally:sys.argv=original_argv;ev.load_hybrid_environment=original_load
    record=json.loads(Path(job['result_path']).read_text())
    if len(rows)!=job['task']['steps']:raise ValueError('incomplete allocation audit')
    cooling=[v for r in rows for v in r['cooling'].values()];heating=[v for r in rows for v in r['heating']]
    summary=dict(steps=len(rows),thermal_model='Offline current-state static heat calculation; implicit within-step dynamics and delayed public estimates are not preserved.',
        cooling_candidate_handler_steps=len(cooling),fan_Wh=sum(r['fan_Wh'] for r in rows),
        cooling_Wh=sum(r['cooling_Wh'] for r in rows),gas_Wh=sum(r['gas_Wh'] for r in rows),
        static_candidate_fan_saving_Wh=sum(v['fan_saving_Wh'] for v in cooling),
        static_candidate_cooling_saving_Wh=sum(v['cooling_saving_Wh'] for v in cooling),
        heated_zone_steps=len(heating),heat_weighted_ideal_flow_reduction=
        sum(v['net_heat_W']*v['ideal_flow_reduction'] for v in heating)/max(1e-12,sum(v['net_heat_W'] for v in heating)),
        assumption='Cooling target is2K lower clipped at285K; no active reheat; all room static cooling requests preserved; minimum flow and AHU pressure constraints enforced. Heating is an optimistic bound with unlimited valve capacity, not a feasible controller.',
        controller_unchanged=True,reward_info_unchanged=True,per_device_system_meters_reconciled=True,
        exact_v10_shadow_steps=record['shadow_action_parity_steps'])
    write_json(Path(job['result_path']).with_suffix('.allocation.json'),dict(summary=summary,rows=rows,fingerprint=job['fingerprint']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
