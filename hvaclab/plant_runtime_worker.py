"""Non-intervening plant telemetry on cached-matched V10 and full-feedback traces."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.phase_online_worker import verify_record as verify_base
from hvaclab.plant_runtime_telemetry import stage_taps,snapshot,metered_watts,aggregate


def verify_record(record,job):
    verify_base(record,job)
    data=record['plant_telemetry'];series=data['series']
    if len(series)!=record['result']['steps']:raise ValueError('incomplete plant telemetry')
    if data['phase_totals']!=aggregate(series):raise ValueError('plant aggregate changed')
    if data['actual_command_mismatch_steps']!=0:raise ValueError('device command/state mismatch')
    if job.get('reference'):
        old=json.loads(Path(job['reference']['path']).read_text())
        for phase,values in old['phase_sums'].items():
            for field,value in values.items():
                if not np.isclose(record['phase_sums'][phase][field],value,rtol=0,atol=1e-8):raise ValueError('phase instrumentation changed '+phase+'/'+field)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);args=parser.parse_args()
    path=Path(args.job);job=json.loads(path.read_text())
    for source,sha in job['source_sha256'].items():
        if digest(source)!=sha:raise ValueError('source changed '+source)
    from smart_buildings import evaluate as ev
    from hvaclab import phase_online_worker as base
    init=ev.initialize_policy;load=ev.load_hybrid_environment;state={};series=[];transitions=[]
    def primary():return state.get('policy')
    def initialize(cls,meta,spec):
        policy=init(cls,meta,spec)
        if hasattr(policy,'feedback_phase'):
            state.update(policy=policy,last_run=[0]*len(policy.ahu_controllers),last_pump=0,
                         last_refresh=[None]*len(policy.ahu_controllers),previous_phase=None)
            action=policy.get_action
            def capture(obs):
                command=action(obs);phase=policy.decision_phase(obs);t=len(series);n=len(policy.ahu_controllers)
                if t==0:
                    hvac=state['env'].building.simulator.hvac
                    devices={d.device_id():d for d in getattr(hvac.air_handler,'ahus',[hvac.air_handler])}
                    state['last_run']=[int(devices[c['ahu_id']].run_command) for c in policy.ahu_controllers]
                    state['last_pump']=int(hvac.hot_water_system.run_command)
                for i,r in enumerate(frame['refreshed']):
                    if r:state['last_refresh'][i]=dict(step=t,phase=phase)
                run=snapshot(policy,command)['run'];pump=int(command['discrete_action'][policy.water_discrete_idx])
                for i,on in enumerate(run):
                    if int(bool(frame['parent']['run'][i]))+int(frame['hold'][i])+int(frame['heat_override'][i])!=on:
                        raise AssertionError('AHU running reason not exclusive/exhaustive')
                valve=[any(command['continuous_action'][indices['reheat_valve_percentage_command']]>-1+1e-6 for _,_,indices in c['zones']) for c in policy.ahu_controllers]
                pending=dict(step=t,phase=phase,run=run,pump=pump,parent_run=list(frame['parent']['run']),
                    parent_sat=list(frame['parent']['sat']),hold=list(frame['hold']),heat_override=list(frame['heat_override']),
                    refreshed=list(frame['refreshed']),new_hold=list(frame['new_hold']),hold_left=list(frame['hold_after']),
                    hold_origin=[x['phase'] if x else None for x in state['last_refresh']],
                    last_cooling_request=copy.deepcopy(state['last_refresh']),
                    held_reheat=[bool(frame['hold'][i] and valve[i]) for i in range(n)],
                    open_valves=sum(command['continuous_action'][indices['reheat_valve_percentage_command']]>-1+1e-6 for c in policy.ahu_controllers for _,_,indices in c['zones']),
                    starts=[int(on and not prev) for on,prev in zip(run,state['last_run'])],pump_start=int(pump and not state['last_pump']),
                    stops=[int(prev and not on) for on,prev in zip(run,state['last_run'])],pump_stop=int(state['last_pump'] and not pump))
                pending['open_valves']=int(pending['open_valves'])
                if phase!=state['previous_phase']:transitions.append(dict(step=t,from_phase=state['previous_phase'],to_phase=phase,
                    hold_left=pending['hold_left'],last_cooling_request=pending['last_cooling_request'],run=run,pump=pump))
                state.update(pending=pending,last_run=run.copy(),last_pump=pump,previous_phase=phase)
                return command
            policy.get_action=capture
        return policy
    def load_environment(*a,**kw):
        env=load(*a,**kw);state['env']=env;reward=env.get_reward_info_and_response
        def record_reward():
            info,response=reward()
            pending=state.pop('pending',None)
            if pending is not None:
                hvac=env.building.simulator.hvac
                ahus=getattr(hvac.air_handler,'ahus',[hvac.air_handler]);devices={d.device_id():d for d in ahus}
                actual=[int(devices[c['ahu_id']].run_command) for c in primary().ahu_controllers]
                actual_pump=int(hvac.hot_water_system.run_command)
                if actual!=pending['run'] or actual_pump!=pending['pump']:raise AssertionError('actual device state differs from command')
                dt=info.end_timestamp.seconds-info.start_timestamp.seconds+(info.end_timestamp.nanos-info.start_timestamp.nanos)/1e9
                if dt<=0:raise ValueError('invalid meter duration')
                pending.update(dt_seconds=dt,watts=metered_watts(info),actual_run=actual,actual_pump=actual_pump,
                    weighted_energy=.2*float(response.normalized_energy_cost),weighted_carbon=.2*float(response.normalized_carbon_emission))
                series.append(pending)
            return info,response
        env.get_reward_info_and_response=record_reward;return env
    core_job=copy.deepcopy(job);final_path=Path(job['result_path']);core_path=final_path.with_suffix('.core.json')
    core_job['result_path']=str(core_path);core_descriptor=path.with_suffix('.corejob.json');write_json(core_descriptor,core_job)
    old_argv=sys.argv;ev.initialize_policy=initialize;ev.load_hybrid_environment=load_environment
    try:
        with stage_taps(primary) as frame:
            sys.argv=[old_argv[0],'--job',str(core_descriptor)];base.main()
    finally:sys.argv=old_argv;ev.initialize_policy=init;ev.load_hybrid_environment=load
    record=json.loads(core_path.read_text());record['plant_telemetry']=dict(series=series,transitions=transitions,
        phase_totals=aggregate(series),actual_command_mismatch_steps=0,ahu_ids=[c['ahu_id'] for c in primary().ahu_controllers],
        semantics='offline observer only; per-step powers are scorer-metered W integrated over reward intervals, not continuous physical meter samples')
    verify_record(record,job);write_json(final_path,record)
    print('PLANT_TELEMETRY_VERIFIED',len(series),flush=True)


if __name__=='__main__':main()
