"""Isolated short rollout, causal learner diagnostics and exact V10 shadow control."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import write_json,digest,valid_result


def register():
    from hvaclab.pipeline_entry import register as parents
    from hvaclab.registry import _VARIANTS,_OURS
    parents();_OURS['online_actuator']=('hvaclab.policies.online_actuator','OnlineActuatorPolicy')
    for key,blend in (('online_shadow',0.),('online_quarter',.25)):
        _VARIANTS[key]=('online_actuator',{**_VARIANTS['damper12_mid'][1],
            'online_blend':blend,'online_limit_k':.5,'online_warmup':288})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);args=parser.parse_args()
    path=Path(args.job);job=json.loads(path.read_text());task=job['task']
    for source,sha in job['source_sha256'].items():
        if digest(source)!=sha:raise ValueError('frozen source changed: '+source)
    register()
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy
    from hvaclab.runner import _run_one
    original=ev.initialize_policy;captured=[];parity=[0]
    def initialize(cls,meta,spec):
        policy=original(cls,meta,spec);captured.append(policy)
        if task['policy']=='online_shadow':
            baseline=original(resolve_policy('damper12_mid'),meta,spec);get_action=policy.get_action
            def compare(obs):
                candidate=get_action(obs);reference=baseline.get_action(obs)
                for field in candidate:
                    if not np.array_equal(candidate[field],reference[field]):raise AssertionError('online shadow V10 parity failure: '+field)
                parity[0]+=1;return candidate
            policy.get_action=compare
        return policy
    ev.initialize_policy=initialize
    try:row=_run_one((task['building'],task['weather'],task['policy'],task['steps'],task['scenarios_dir']))
    finally:ev.initialize_policy=original
    diagnostics=captured[0].online_report() if captured else {}
    write_json(job['result_path'],dict(task=task,fingerprint=job['fingerprint'],result=row,
        online_diagnostics=diagnostics,shadow_action_parity_steps=parity[0]))
    if not valid_result(row,task):raise SystemExit(1)
    if task['policy']=='online_shadow' and parity[0]!=task['steps']:raise SystemExit('incomplete exact parity')
    print('DONE',task['building'],task['weather'],task['policy'],abs(row['mean_reward']),flush=True)


if __name__=='__main__':main()
