"""Short rollouts with conserved reward decomposition by decision-time phase."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import write_json,digest,valid_result
from hvaclab.policies.phase_online_actuator import PHASES


def register():
    from hvaclab.pipeline_entry import register as parents
    from hvaclab.registry import _OURS,_VARIANTS
    parents();_OURS['phase_online']=('hvaclab.policies.phase_online_actuator','PhaseOnlineActuatorPolicy')
    for key,phase,blend in (('phase_ctrl','all',0.),('phase_all','all',.25),('phase_prestart','prestart',.25),('phase_occupied','occupied',.25)):
        _VARIANTS[key]=('phase_online',{**_VARIANTS['damper12_mid'][1],'online_blend':blend,
            'online_limit_k':.5,'online_warmup':288,'feedback_phase':phase})


def verify_record(record,job):
    task=job['task'];row=record['result'];buckets=record['phase_sums'];parity=[record['shadow_action_parity_steps']]
    if record['task']!=task or record['fingerprint']!=job['fingerprint']:raise ValueError('record identity changed')
    if not valid_result(row,task):raise ValueError('invalid rollout')
    for source,total in (('reward_sum','total_reward'),('prod_regret','prod_regret'),('energy_cost','energy_cost'),('carbon','carbon')):
        if not np.isclose(sum(b[source] for b in buckets.values()),row[total],rtol=1e-10,atol=1e-9):raise ValueError('phase totals do not conserve '+source)
    if sum(b['steps'] for b in buckets.values())!=row['steps']:raise ValueError('phase step count mismatch')
    if task['policy']=='phase_ctrl' and parity[0]!=row['steps']:raise ValueError('incomplete V10 action parity')
    if job.get('reference'):
        ref=job['reference']
        if digest(ref['path'])!=ref['sha256']:raise ValueError('reference changed')
        previous=json.loads(Path(ref['path']).read_text())['result']
        for key in ('steps','total_reward','prod_regret','energy_cost','carbon'):
            if not np.isclose(row[key],previous[key],rtol=0,atol=1e-8):raise ValueError('instrumented reference mismatch '+key)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);args=parser.parse_args()
    job=json.loads(Path(args.job).read_text());task=job['task']
    for source,sha in job['source_sha256'].items():
        if digest(source)!=sha:raise ValueError('frozen input changed: '+source)
    register()
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy
    from hvaclab.runner import _run_one
    init=ev.initialize_policy;load=ev.load_hybrid_environment;captured=[];phase=['offhours'];parity=[0]
    buckets={p:dict(steps=0,reward_occupied_steps=0,reward_sum=0.,prod_regret=0.,energy_cost=0.,carbon=0.) for p in PHASES}
    def initialize(cls,meta,spec):
        policy=init(cls,meta,spec);captured.append(policy);action=policy.get_action
        baseline=init(resolve_policy('damper12_mid'),meta,spec) if task['policy']=='phase_ctrl' else None
        def get_action(obs):
            phase[0]=policy.decision_phase(obs);command=action(obs)
            if baseline is not None:
                reference=baseline.get_action(obs)
                for key in command:
                    if not np.array_equal(command[key],reference[key]):raise AssertionError('V10 phase-control action mismatch')
                parity[0]+=1
            return command
        policy.get_action=get_action;return policy
    def load_environment(*args,**kwargs):
        env=load(*args,**kwargs);step=env.step
        def step_and_record(action):
            result=step(action);_,rr=env.get_reward_info_and_response();b=buckets[phase[0]]
            b['steps']+=1;b['reward_occupied_steps']+=int(rr.total_occupancy>0);b['reward_sum']+=float(result[1])
            b['prod_regret']+=float(rr.normalized_productivity_regret)
            b['energy_cost']+=float(rr.normalized_energy_cost);b['carbon']+=float(rr.normalized_carbon_emission)
            if sum(x['steps'] for x in buckets.values())%288==0:print('HEARTBEAT',sum(x['steps'] for x in buckets.values()),flush=True)
            return result
        env.step=step_and_record;return env
    ev.initialize_policy=initialize;ev.load_hybrid_environment=load_environment
    try:row=_run_one((task['building'],task['weather'],task['policy'],task['steps'],task['scenarios_dir']))
    finally:ev.initialize_policy=init;ev.load_hybrid_environment=load
    diagnostics=captured[0].online_report() if captured else {}
    record=dict(task=task,fingerprint=job['fingerprint'],result=row,phase_sums=buckets,
                online_diagnostics=diagnostics,shadow_action_parity_steps=parity[0])
    write_json(job['result_path'],record)
    verify_record(record,job)
    print('DONE',task['building'],task['policy'],abs(row['mean_reward']),flush=True)


if __name__=='__main__':main()
