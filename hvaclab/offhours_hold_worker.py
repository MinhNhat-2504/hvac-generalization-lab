"""Isolated hold-refresh intervention with daily first-occupied comfort checks."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from hvaclab.coverage import digest,write_json
from hvaclab.phase_online_worker import verify_record as verify_base
from hvaclab.plant_runtime_telemetry import stage_taps


def register():
    from hvaclab.pipeline_entry import register as parents
    from hvaclab.registry import _OURS,_VARIANTS
    parents();_OURS['offhours_hold']=('hvaclab.policies.offhours_hold','OffhoursHoldPolicy')
    for key,cap in (('phase_ctrl',None),('hold_refresh6',6),('hold_refresh24',24)):
        _VARIANTS[key]=('offhours_hold',{**_VARIANTS['damper12_mid'][1],'offhours_hold_steps':cap})


class MorningAccumulator:
    def __init__(self,time_zone):self.time_zone=time_zone;self.days={}
    def add(self,epoch,response):
        clock=pd.Timestamp(epoch,unit='s',tz='UTC').tz_convert(self.time_zone);day=str(clock.date())
        if day not in self.days and response.total_occupancy>0:
            self.days[day]=dict(first_occupied_epoch=epoch,first_local_clock=str(clock),steps=0,occupied_steps=0,
                reward_sum=0.,prod_regret=0.,energy_cost=0.,carbon=0.,worst_occupied_comfort=0.)
        if day not in self.days:return
        row=self.days[day]
        if not 0<=epoch-row['first_occupied_epoch']<7200:return
        row['steps']+=1;row['occupied_steps']+=int(response.total_occupancy>0)
        row['reward_sum']+=float(response.agent_reward_value);row['prod_regret']+=float(response.normalized_productivity_regret)
        row['energy_cost']+=float(response.normalized_energy_cost);row['carbon']+=float(response.normalized_carbon_emission)
        if response.total_occupancy>0:row['worst_occupied_comfort']=max(row['worst_occupied_comfort'],-.6*float(response.normalized_productivity_regret))


def verify_record(record,job):
    verify_base(record,job)
    morning=record['morning_comfort'];n=sum(x['steps'] for x in morning.values())
    if not morning or n<=0:raise ValueError('no morning exposure')
    if record['hold_diagnostics']['normal_hold_steps']!=432:raise ValueError('normal V10 hold changed')
    if record['hold_audit']['protected_capped_refreshes']!=0:raise ValueError('limited hold in protected phase')
    for row in morning.values():
        if not 1<=row['steps']<=24:raise ValueError('invalid first-two-hour window')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);args=parser.parse_args()
    path=Path(args.job);job=json.loads(path.read_text())
    for p,sha in job['source_sha256'].items():
        if digest(p)!=sha:raise ValueError('source changed '+p)
    from smart_buildings import evaluate as ev
    from hvaclab import phase_online_worker as base
    init=ev.initialize_policy;load=ev.load_hybrid_environment;old_register=base.register
    state={};audit=dict(capped_refreshes=0,protected_capped_refreshes=0,carried_above_cap_handler_steps=0,
                       hold_handler_steps=0,offhours_hold_handler_steps=0,offhours_ahu_handler_steps=0,offhours_pump_steps=0)
    def primary():return state.get('policy')
    def initialize(cls,meta,spec):
        policy=init(cls,meta,spec)
        if hasattr(policy,'hold_report'):
            state['policy']=policy;state['morning']=MorningAccumulator(meta['time_zone']);get_action=policy.get_action
            policy.online_report=policy.hold_report
            def action(obs):
                result=get_action(obs);phase=policy.decision_phase(obs)
                capped=policy.last_limited
                if capped:
                    audit['capped_refreshes']+=sum(frame['refreshed'])
                    audit['protected_capped_refreshes']+=int(phase!='offhours')*sum(frame['refreshed'])
                    for refresh,left in zip(frame['refreshed'],frame['hold_after']):
                        if refresh and left!=policy.last_effective_hold_steps:raise AssertionError('refresh duration mismatch')
                    audit['carried_above_cap_handler_steps']+=sum(left>policy.last_effective_hold_steps for left in frame['hold_after'])
                audit['hold_handler_steps']+=sum(frame['hold'])
                if phase=='offhours':
                    audit['offhours_hold_handler_steps']+=sum(frame['hold'])
                    audit['offhours_ahu_handler_steps']+=sum(int(result['discrete_action'][c['run_discrete_idx']]) for c in policy.ahu_controllers)
                    audit['offhours_pump_steps']+=int(result['discrete_action'][policy.water_discrete_idx])
                state['pending']=True;return result
            policy.get_action=action
        return policy
    def load_environment(*a,**kw):
        env=load(*a,**kw);reward=env.get_reward_info_and_response
        def capture():
            info,response=reward()
            if state.pop('pending',False):
                epoch=info.start_timestamp.seconds+info.start_timestamp.nanos/1e9
                state['morning'].add(epoch,response)
            return info,response
        env.get_reward_info_and_response=capture;return env
    final_path=Path(job['result_path']);core=copy.deepcopy(job);core['result_path']=str(final_path.with_suffix('.core.json'))
    descriptor=path.with_suffix('.corejob.json');write_json(descriptor,core);argv=sys.argv
    ev.initialize_policy=initialize;ev.load_hybrid_environment=load_environment;base.register=register
    try:
        with stage_taps(primary) as frame:
            sys.argv=[argv[0],'--job',str(descriptor)];base.main()
    finally:ev.initialize_policy=init;ev.load_hybrid_environment=load;base.register=old_register;sys.argv=argv
    record=json.loads(Path(core['result_path']).read_text());record['hold_diagnostics']=record.pop('online_diagnostics')
    record.update(morning_comfort=state['morning'].days,hold_audit={k:int(v) for k,v in audit.items()})
    verify_record(record,job);write_json(final_path,record);print('MORNING_AND_HOLD_AUDIT_VERIFIED',flush=True)


if __name__=='__main__':main()
