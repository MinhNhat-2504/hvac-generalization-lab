"""Development-only matched-state 2x2 PI-integral/hold memory intervention."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

CASES={'singapore':('building_9002','singapore_2','2023-03-20'),
       'atlanta':('building_9001','atlanta_4','2023-06-01')}


def intervene(policy,reset_pi=False,reset_hold=False):
    """One-time state intervention; no gains, projection or plant edits."""
    if reset_pi:policy._zone_integral.clear()
    if reset_hold:
        policy._hold_left=[0]*len(policy._hold_left)
        policy._held_action=[None]*len(policy._held_action)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,default='singapore')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--plan',type=Path,help='Portable private branch plan, prepared from existing development records.')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];os.chdir(repo)
    sys.path[:0]=[str(repo),str(Path(__file__).resolve().parent)]
    import numpy as np
    import psutil
    from probe_paper_branches import numba_numpy_state
    from smart_buildings import evaluate as ev
    from hvaclab.pipeline_entry import register
    from hvaclab.registry import _VARIANTS
    from hvaclab.zone_reserve_policy import ZoneReservePolicy
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=args.output.resolve()
    if not root.is_relative_to(repo/'experiments/runs'):raise ValueError('output outside run root')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():raise ValueError('preserve original results')
    def write(name,data):(root/name).write_text(json.dumps(data,indent=2)+'\n')
    if psutil.virtual_memory().available<1.75*2**30:
        write('waiting.json',dict(status='waiting_for_memory',minimum_gib=1.75));print('WAITING_FOR_MEMORY');return
    frozen=json.loads(Path('experiments/runs/paper_confirmation_20261106/protocol.json').read_text())['source_sha256']
    if any(sha(n)!=h for n,h in frozen.items()):raise ValueError('scientific source changed')
    b,w,day=CASES[args.case];prefix=Path('experiments/runs/reserve_lift_short/jobs')/(b+'__'+w+'__reserve20')
    if args.plan:
        selected=json.loads(args.plan.read_text())['cases'][args.case]
        if selected['case']!=b+'/'+w or selected['date']!=day:raise ValueError('portable plan case differs')
        epoch=selected['branch_epoch'];warmup=selected['branch_step'];expected_morning=selected['reference_morning_comfort']
        reference_sha=selected['reference_sha256'];trace_sha=selected['trace_sha256']
    else:
        old=json.loads(Path(str(prefix)+'.result.json').read_text())
        thermal=json.loads(Path(str(prefix)+'.result.thermal.json').read_text())
        epoch=old['morning_comfort'][day]['first_occupied_epoch']
        matches=[int(x['step']) for x in thermal['rows'] if x['epoch']==epoch]
        if len(matches)!=1:raise ValueError('branch epoch ambiguous')
        warmup=matches[0];expected_morning=old['morning_comfort'][day]['prod_regret']*(-.6)/24
        reference_sha=sha(str(prefix)+'.result.json');trace_sha=sha(str(prefix)+'.result.thermal.json')
        del thermal;gc.collect()
    if not isinstance(warmup,int) or not 1<=warmup<=4032:raise ValueError('invalid branch step')
    horizon=144
    protocol=dict(case=b+'/'+w,date=day,branch_step=warmup,branch_epoch=epoch,horizon=horizon,
        selection='Earliest failed morning in the existing reserve20 development prefix for this case; post-selected mechanistic diagnosis, not confirmation.',
        warmup_policy='reserve20',arms=['sham','control','pi_integral_reset','hold_memory_reset','both_reset'],
        intervention='One-time zeroing at branch only; normal integration and hold refresh immediately resume.',
        script_sha256=sha(__file__),reference_sha256=reference_sha,trace_sha256=trace_sha,
        preserved_frozen_source_sha256=frozen,scope='Matched plant/controller history except declared state edit. Exploratory, not a proposed deployment fix or organizer score.')
    write('protocol.json',protocol)
    register();path=Path('smart_buildings/holdout_scenarios/dev')/b/w
    env=ev.load_hybrid_environment(path/'config.gin');meta=ev.create_building_meta_data(env,path)
    policy=ZoneReservePolicy(meta,env.action_spec(),**_VARIANTS['damper12_mid'][1],flow_equivalent=True,
        cooling_lift=True,project_lift=True,comfort_guard=False,reserve_fraction=.2)
    obs,_=env.reset();started=time.monotonic()
    for i in range(warmup):
        action=ev.validate_action(policy.get_action(obs),env.action_spec());obs,_,end,trunc,_=env.step(action)
        if end or trunc:raise ValueError('unexpected warmup end')
        if (i+1)%144==0:print('WARMUP',i+1,'/',warmup,flush=True)
    py_state=random.getstate();np_state=np.random.get_state();compiled=numba_numpy_state()
    initial=dict(integral=dict(policy._zone_integral),signed_pi=dict(policy._zone_u),hold=list(policy._hold_left),
        held_action=list(policy._held_action),projection_history=dict(policy._previous_lift_temperatures))
    write('branch_controller_state.json',initial)
    def clone():
        factor=env.building.simulator._solver._factorization
        e,p,o=copy.deepcopy((env,policy,obs),{id(factor):factor} if factor is not None else {})
        for x,y in ((env.building.simulator.building.temp,e.building.simulator.building.temp),
                    (env.building.simulator._solver._mass_temp,e.building.simulator._solver._mass_temp)):
            if np.shares_memory(x,y) or not np.array_equal(x,y):raise ValueError('plant copy not independent/equal')
        return e,p,o
    def run(label,reset_pi=False,reset_hold=False):
        e,p,o=clone();intervene(p,reset_pi,reset_hold)
        random.setstate(py_state);np.random.set_state(np_state);numba_numpy_state(compiled)
        rows=[]
        for i in range(horizon):
            action=ev.validate_action(p.get_action(o),e.action_spec());o,reward,end,trunc,_=e.step(action)
            info,response=e.get_reward_info_and_response();sim=e.building.simulator
            when=info.start_timestamp.seconds+info.start_timestamp.nanos/1e9
            if when!=epoch+i*300:raise ValueError('branch clock differs')
            rows.append(dict(step=i,epoch=when,reward=float(reward),comfort=-.6*float(response.normalized_productivity_regret),
                energy=.2*float(response.normalized_energy_cost),carbon=.2*float(response.normalized_carbon_emission),
                action={k:np.asarray(v).tolist() for k,v in action.items()},
                obs_sha256=hashlib.sha256(np.asarray(o).tobytes()).hexdigest(),
                grid_sha256=hashlib.sha256(sim.building.temp.tobytes()).hexdigest(),
                wall_sha256=hashlib.sha256(sim._solver._mass_temp.tobytes()).hexdigest(),
                integral=dict(p._zone_integral),hold=list(p._hold_left)))
            if end or trunc:raise ValueError('unexpected continuation end')
        write(label+'.json',rows);del e,p,o;gc.collect();print('ARM_COMPLETE',label,flush=True);return rows
    sham=run('sham');control=run('control')
    if sham!=control:
        write('result.json',dict(status='sham_replay_failed',no_intervention_run=True));raise ValueError('identity gate failed')
    first=np.mean([x['comfort'] for x in control[:24]])
    if not np.isclose(first,expected_morning,rtol=0,atol=1e-9):
        write('result.json',dict(status='development_reference_mismatch',no_intervention_run=True,observed=first));raise ValueError('reference mismatch')
    arms={'control':control}
    for name,pi,hold in (('pi_integral_reset',True,False),('hold_memory_reset',False,True),('both_reset',True,True)):
        arms[name]=run(name,pi,hold)
    metrics={name:dict(morning_comfort=float(np.mean([x['comfort'] for x in rows[:24]])),
        loss_12h=-float(np.mean([x['reward'] for x in rows])),comfort_12h=float(np.mean([x['comfort'] for x in rows])),
        first_different_action=next((i for i,(a,c) in enumerate(zip(rows,control)) if a['action']!=c['action']),None)) for name,rows in arms.items()}
    interaction=metrics['both_reset']['morning_comfort']-metrics['pi_integral_reset']['morning_comfort']-metrics['hold_memory_reset']['morning_comfort']+metrics['control']['morning_comfort']
    if any(sha(n)!=h for n,h in frozen.items()):raise ValueError('scientific source changed')
    write('result.json',dict(status='matched_state_factorial_verified',protocol_sha256=sha(root/'protocol.json'),
        sham_exact_steps=horizon,reference_morning_reproduced=True,metrics=metrics,
        morning_interaction=interaction,seconds=time.monotonic()-started,
        intervention_scope=protocol['scope'],notebook_exported=False))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
