"""Short, local common-state branching feasibility probe; not a challenge score.

Same-process deepcopy may share a native immutable linear-solver factorization,
but dynamic arrays, controller history and RNG state must be independent.
Identical continuation must match before any diagnostic intervention runs.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time


def array_difference(left, right):
    """Describe numerical drift without relaxing the exact replay gate."""
    import numpy as np
    a=np.asarray(left);b=np.asarray(right)
    if a.shape!=b.shape:raise ValueError('state shape mismatch')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('nonfinite branch state')
    return dict(exact=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)),
                max_absolute=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0)))


def numba_numpy_state(state=None):
    """Version-sensitive, same-thread Numba RNG snapshot/restore for this probe."""
    from numba import _helperlib
    pointer=_helperlib.rnd_get_np_state_ptr()
    if state is None:return copy.deepcopy(_helperlib.rnd_get_state(pointer))
    _helperlib.rnd_set_state(pointer,copy.deepcopy(state))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup',type=int,default=432)
    parser.add_argument('--continuation',type=int,default=48)
    parser.add_argument('--with-interventions',action='store_true')
    parser.add_argument('--output',default='experiments/runs/common_state_probe')
    parser.add_argument('--restore-numba-rng',action='store_true',
                        help='Restore compiled convection RNG too; private API, single-thread only.')
    args=parser.parse_args()
    if not 1<=args.warmup<=576 or not 1<=args.continuation<=144:raise ValueError('short probe only')
    repo=Path(__file__).resolve().parents[2];os.chdir(repo);sys.path.insert(0,str(repo))
    import numpy as np
    import numba
    import psutil
    if psutil.virtual_memory().available<1.75*2**30:raise RuntimeError('probe needs1.75GiB available; do not oversubscribe')
    from smart_buildings import evaluate as ev
    from hvaclab.pipeline_entry import register
    from hvaclab.registry import _VARIANTS,resolve_policy
    from hvaclab.zone_reserve_policy import ZoneReservePolicy
    from hvaclab.coverage import write_json,digest
    register();root=Path(args.output).resolve()
    if not root.is_relative_to(repo/'experiments/runs'):raise ValueError('output must stay in experiments/runs')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():raise ValueError('probe already exists; preserve original result')
    frozen={p.as_posix():digest(p) for p in Path('hvaclab').rglob('*.py')}
    frozen.update({p.as_posix():digest(p) for p in Path('smart_buildings').rglob('*.py')})
    write_json(root/'protocol.json',dict(warmup=args.warmup,continuation=args.continuation,
        source_sha256=frozen,probe_sha256=digest(Path(__file__)),case='building_9002/singapore_2',
        interventions=args.with_interventions,restore_numba_rng=args.restore_numba_rng,
        numba_version=numba.__version__,scope='Reused exploratory case, common V10 warm-up, same-process copy; no causal attribution to a particular PI/hold component or independent generalization.'))
    env=ev.load_hybrid_environment(Path('smart_buildings/holdout_scenarios/dev/building_9002/singapore_2/config.gin'))
    meta=ev.create_building_meta_data(env,Path('smart_buildings/holdout_scenarios/dev/building_9002/singapore_2'))
    spec=env.action_spec()
    policy=ZoneReservePolicy(meta,spec,**_VARIANTS['damper12_mid'][1],
        flow_equivalent=False,cooling_lift=False,project_lift=True,comfort_guard=False,reserve_fraction=0.)
    shadow=ev.initialize_policy(resolve_policy('damper12_mid'),meta,spec)
    obs,_=env.reset();started=time.monotonic()
    for i in range(args.warmup):
        action=ev.validate_action(policy.get_action(obs),spec);reference=ev.validate_action(shadow.get_action(obs),spec)
        if any(not np.array_equal(action[k],reference[k]) for k in action):raise ValueError('passive projection changed V10 warm-up')
        obs,_,terminated,truncated,_=env.step(action)
        if terminated or truncated:raise ValueError('unexpected termination during warm-up')
        if (i+1)%144==0:print('WARMUP',i+1,flush=True)
    rng=(random.getstate(),np.random.get_state())
    compiled_rng=numba_numpy_state() if args.restore_numba_rng else None

    def clone():
        factor=env.building.simulator._solver._factorization
        # SuperLU cannot generally be pickled. Reuse only this numerical
        # factorization; mutable thermal/controller state is deep-copied.
        e,p,o=copy.deepcopy((env,policy,obs),{id(factor):factor} if factor is not None else {})
        for a,b in ((env.building.simulator.building.temp,e.building.simulator.building.temp),
                    (env.building.simulator._solver._mass_temp,e.building.simulator._solver._mass_temp)):
            if np.shares_memory(a,b):raise ValueError('thermal state is aliased')
        if p._zone_integral is policy._zone_integral or p._hold_left is policy._hold_left:
            raise ValueError('controller state is aliased')
        return e,p,o

    def run(e,p,o,label,shadow_policy=None):
        random.setstate(rng[0]);np.random.set_state(rng[1]);rows=[]
        if compiled_rng is not None:numba_numpy_state(compiled_rng)
        for i in range(args.continuation):
            action=ev.validate_action(p.get_action(o),e.action_spec())
            if shadow_policy is not None:
                expected=ev.validate_action(shadow_policy.get_action(o),e.action_spec())
                if any(not np.array_equal(action[k],expected[k]) for k in action):raise ValueError('reference continuation differs from V10')
            action_hash=hashlib.sha256(b''.join(np.asarray(action[k]).tobytes() for k in sorted(action))).hexdigest()
            o,reward,terminated,truncated,_=e.step(action);info,response=e.get_reward_info_and_response()
            sim=e.building.simulator
            if i==0:
                np.savez_compressed(root/(label+'_first_state.npz'),
                    thermal=sim.building.temp,mass=sim._solver._mass_temp)
            rows.append(dict(step=i,action_sha256=action_hash,observation_sha256=hashlib.sha256(np.asarray(o).tobytes()).hexdigest(),
                observation=np.asarray(o).tolist(),action={k:np.asarray(v).tolist() for k,v in action.items()},
                reward=float(reward),reward_info_sha256=hashlib.sha256(info.SerializeToString(deterministic=True)).hexdigest(),
                reward_response_sha256=hashlib.sha256(response.SerializeToString(deterministic=True)).hexdigest(),
                thermal_grid_sha256=hashlib.sha256(sim.building.temp.tobytes()).hexdigest(),
                wall_mass_sha256=hashlib.sha256(sim._solver._mass_temp.tobytes()).hexdigest(),
                integral=dict(p._zone_integral),hold=list(p._hold_left),
                weighted_comfort=-.6*float(response.normalized_productivity_regret),
                weighted_energy=.2*float(response.normalized_energy_cost),weighted_carbon=.2*float(response.normalized_carbon_emission)))
            if terminated or truncated:raise ValueError('unexpected continuation end')
        return rows

    # Both copies start before either continuation mutates a state/global RNG.
    try:
        sham_env,sham_policy,sham_obs=clone()
    except BaseException as error:
        write_json(root/'result.json',dict(status='clone_failed',error=repr(error),no_intervention_run=True));raise
    sham=run(sham_env,sham_policy,sham_obs,'sham',copy.deepcopy(shadow))
    write_json(root/'sham.json',sham);del sham_env,sham_policy,sham_obs;gc.collect()
    # Keep the original environment/controller at the common branch point.
    ref_env,ref_policy,ref_obs=clone();reference=run(ref_env,ref_policy,ref_obs,'reference',copy.deepcopy(shadow))
    write_json(root/'reference.json',reference);del ref_env,ref_policy,ref_obs;gc.collect()
    if reference!=sham:
        different=[i for i,(a,b) in enumerate(zip(reference,sham)) if a!=b]
        with np.load(root/'sham_first_state.npz') as a,np.load(root/'reference_first_state.npz') as b:
            state_drift={k:array_difference(a[k],b[k]) for k in a.files}
        action_drift={k:array_difference([x['action'][k] for x in sham],[x['action'][k] for x in reference]) for k in sham[0]['action']}
        for name,sha in frozen.items():
            if digest(name)!=sha:raise ValueError('source changed during probe')
        write_json(root/'result.json',dict(status='identity_continuation_failed',different_steps=different,
            no_intervention_run=True,first_step_drift=state_drift,action_drift=action_drift,
            observation_drift=array_difference([x['observation'] for x in sham],[x['observation'] for x in reference]),
            reward_drift=array_difference([x['reward'] for x in sham],[x['reward'] for x in reference]),
            warmup_v10_action_parity_steps=args.warmup,seconds=time.monotonic()-started,
            scientific_source_unchanged=True,scope='Feasibility failure only; no causal or intervention result.'))
        raise ValueError('identity continuation parity failed; diagnostics saved, intervention blocked')
    result=dict(status='identity_continuation_verified',warmup_v10_action_parity_steps=args.warmup,
        restore_numba_rng=args.restore_numba_rng,numba_version=numba.__version__,
        continuation_parity_steps=args.continuation,shared_native_cache='SuperLU factorization only; dynamic arrays deep-copied',
        scope='One reused case and short horizon; not a general serialization/checkpoint guarantee.',arms={})
    if args.with_interventions:
        for name,budget in (('project',0.),('reserve20',.2)):
            e,p,o=clone();p.flow_equivalent=True;p.cooling_lift=True;p.reserve_fraction=budget
            rows=run(e,p,o,name);write_json(root/(name+'.json'),rows)
            result['arms'][name]=dict(weighted_comfort_mean=float(np.mean([x['weighted_comfort'] for x in rows])),
                loss_mean=-float(np.mean([x['reward'] for x in rows])),reserve_zone_steps=p.reserve_zone_steps,
                lift_handler_steps=p.lift_handler_steps)
            del e,p,o;gc.collect()
    result.update(seconds=time.monotonic()-started,reference=dict(loss_mean=-float(np.mean([x['reward'] for x in reference])),
        weighted_comfort_mean=float(np.mean([x['weighted_comfort'] for x in reference]))))
    for name,sha in frozen.items():
        if digest(name)!=sha:raise ValueError('source changed during probe')
    write_json(root/'result.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
