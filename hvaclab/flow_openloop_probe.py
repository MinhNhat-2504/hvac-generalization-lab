"""Offline command replay to separate allocation error from feedback amplification."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import gc
import json
from pathlib import Path
import numpy as np
from unittest.mock import patch
from hvaclab.coverage import digest,write_json

ROOT=Path('experiments/runs/flow_comfort_diagnosis')


def main():
    from smart_buildings import evaluate as ev
    from hvaclab.pipeline_entry import register
    from hvaclab.registry import resolve_policy,_VARIANTS
    from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
    from hvaclab.policies.offhours_hold import OffhoursHoldPolicy
    register();directory=Path('smart_buildings/training_scenarios/building_train_7/phoenix_3');steps=600
    commands=[];observed=[];rewards=[];schema={}
    def run(mode):
        env=ev.load_hybrid_environment(directory/'config.gin');meta=ev.create_building_meta_data(env,directory);spec=env.action_spec()
        p=resolve_policy('damper12_mid')(meta,spec) if mode=='reference' else FlowEquivalentPolicy(meta,spec,**_VARIANTS['damper12_mid'][1])
        zones=[i for c in p.ahu_controllers for _,i,_ in c['zones']];obs,_=env.reset();series=[];out_reward=[]
        for i in range(steps):
            series.append([float(p._denormalize_zone_temp(obs[j])) for j in zones])
            if mode=='reference':
                action=p.get_action(obs);commands.append({k:np.array(v,copy=True) for k,v in action.items()})
            elif mode=='replay':action={k:np.array(v,copy=True) for k,v in commands[i].items()}
            else:
                command={k:np.array(v,copy=True) for k,v in commands[i].items()}
                with patch.object(OffhoursHoldPolicy,'get_action',return_value=command):action=p.get_action(obs)
            obs,reward,terminated,truncated,_=env.step(ev.validate_action(action,spec));out_reward.append(float(reward))
            if terminated or truncated:raise ValueError('unexpected short replay')
        if mode=='transformed':schema.update(changed_handler_steps=p.rebalanced_handler_steps,max_relative_flow_error=p.max_relative_flow_error)
        del env,p;gc.collect();return np.array(series),out_reward
    baseline,base_reward=run('reference');replay,replay_reward=run('replay')
    np.testing.assert_array_equal(baseline,replay);np.testing.assert_array_equal(base_reward,replay_reward)
    transformed,transformed_reward=run('transformed')
    old_trace=Path('experiments/runs/flow_comfort_diagnosis/jobs/building_train_7__phoenix_3__phase_ctrl.result.thermal.json')
    trace=json.loads(old_trace.read_text());old=np.array([r['observed_temperature'] for r in trace['rows'][:steps]])
    np.testing.assert_array_equal(baseline,old)
    candidate_path=Path('experiments/runs/flow_equivalent_full/jobs/training__building_train_7__phoenix_3__flow_reset.result.thermal.json')
    candidate=np.array([r['observed_temperature'] for r in json.loads(candidate_path.read_text())['rows'][:steps]])
    delta=np.max(np.abs(transformed-baseline),axis=1);closed=np.max(np.abs(candidate-baseline),axis=1)
    result=dict(status='complete_verified',steps=steps,scenario='building_train_7/phoenix_3',
                baseline_reference_trace_sha256=digest(old_trace),candidate_trace_sha256=digest(candidate_path),
                source_sha256=digest(__file__),deterministic_command_replay_exact=True,baseline_prefix_reproduced=True,
                transformed_openloop_max_temperature_difference=float(delta.max()),
                closedloop_max_temperature_difference=float(closed.max()),
                transformed_openloop_reward_sum=sum(transformed_reward),baseline_reward_sum=sum(base_reward),
                allocation_diagnostics=schema,
                checkpoints={str(i):dict(openloop_max=float(delta[:i].max()),closedloop_max=float(closed[:i].max())) for i in (100,200,400,600)},
                interpretation='Same fixed V10 command sequence except pressure/damper allocation; no controller learns from the altered trajectory. This is a finite-horizon single-case diagnostic, not a validated new controller.')
    write_json(ROOT/'openloop_probe.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
