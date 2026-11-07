"""Frozen 72-episode fresh transfer comparison; no revised controller is claimed.

plan -> smoke -> generate -> run -> review. Never edits the original 192-run source,
protocol or evidence. All three policies are locked before materialization.
"""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_n]='1'
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

RUN=Path('experiments/runs/paper_fresh_transfer_20261108')
SCENARIOS=Path('smart_buildings/holdout_paper_fresh_transfer_20261108')
SEED=2026110900
COUNT=12
WEATHER=('denver_4','singapore_5')
ARMS=('V10','reserve20','request_pressure')
STEPS=8640
BASELINE=dict(request_reset=True,pressure_floor_pa=200.,trim_pa=10.,respond_pa=15.,max_response_pa=37.,request_on=.95,request_off=.85,ignored_requests=0,update_steps=1,startup_delay_steps=1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(canonical(x));temp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def check_hashes(repo,hashes):
    for name,h in hashes.items():
        path=(repo/name).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or sha(path)!=h:raise ValueError('source/input mismatch: '+name)
def expected_tasks(buildings):
    if len(buildings)!=COUNT or len(set(buildings))!=COUNT:raise ValueError('need12unique buildings')
    return [dict(building=b,weather=w,policy=a,steps=STEPS) for b in buildings for w in WEATHER for a in ARMS]
def collect_weather(x,known):
    result=set()
    if isinstance(x,dict):
        for key,value in x.items():
            if key in ('weather','weathers'):
                if isinstance(value,str) and value in known:result.add(value)
                elif isinstance(value,list):result.update(v for v in value if isinstance(v,str) and v in known)
            if key not in ('source_sha256','input_sha256','files_sha256','preserved_frozen_sha256'):
                result.update(collect_weather(value,known))
    elif isinstance(x,list):
        for value in x:result.update(collect_weather(value,known))
    return result

def lock_protocol(repo):
    path=repo/RUN/'protocol.json'
    if path.exists():return load_protocol(repo)
    import yaml
    from hvaclab.pipeline_entry import register
    from hvaclab.registry import _VARIANTS
    register()
    old=repo/'experiments/runs/paper_confirmation_20261106/protocol.json'
    frozen=read(old)['source_sha256'];check_hashes(repo,frozen)
    if (repo/SCENARIOS).exists():raise ValueError('fresh scenario root already exists before lock')
    seeds=[repo/'smart_buildings/buildings'/f'building_{seed:04d}.yaml' for seed in range(SEED,SEED+120)]
    if any(p.exists() for p in seeds):raise ValueError('fresh seed range already materialized')
    known={p.stem:p for p in (repo/'smart_buildings/weather').glob('*.yaml')}
    exposed=set();inventory={}
    for p in sorted(set((repo/'experiments/runs').rglob('protocol.json'))|set((repo/'experiments/runs').rglob('cohort.json'))):
        try:data=read(p)
        except (OSError,ValueError):continue
        exposed.update(collect_weather(data,set(known)));inventory[p.relative_to(repo).as_posix()]=sha(p)
    exposed.update(p.parent.name for p in (repo/'smart_buildings').rglob('config.gin') if p.parent.name in known)
    training=yaml.safe_load((repo/'smart_buildings/training_set.yaml').read_text())
    exposed.update(v['weather'] for v in training['scenarios'])
    if set(WEATHER)&exposed:raise ValueError('proposed fresh strip already exposed')
    windows={}
    for w in WEATHER:
        d=yaml.safe_load(known[w].read_text());start=datetime.fromisoformat(str(d['start_timestamp']));end=start+timedelta(days=30)
        for old_w in exposed:
            if old_w.rsplit('_',1)[0]!=w.rsplit('_',1)[0]:continue
            a=datetime.fromisoformat(str(yaml.safe_load(known[old_w].read_text())['start_timestamp']));b=a+timedelta(days=30)
            if max(a,start)<min(b,end):raise ValueError('new strip overlaps exposed weather window')
        windows[w]=dict(start=start.isoformat(),end_exclusive=end.isoformat(),sha256=sha(known[w]))
    added={Path(__file__).resolve().relative_to(repo).as_posix():sha(__file__),
           'reports/scripts/request_pressure_baseline.py':sha(repo/'reports/scripts/request_pressure_baseline.py')}
    protocol=dict(status='policies_locked_before_fresh_generation',buildings=COUNT,seed_start=SEED,max_seed_attempts=120,
        weather=list(WEATHER),weather_windows=windows,arms=list(ARMS),steps=STEPS,total_jobs=72,
        v10_parameters=_VARIANTS['damper12_mid'][1],request_parameters=BASELINE,
        reserve_parameters=dict(flow_equivalent=True,cooling_lift=True,project_lift=True,comfort_guard=False,reserve_fraction=.2),
        preserved_frozen_sha256=frozen,additional_source_sha256=added,original_confirmation_protocol_sha256=sha(old),
        observed_exposure_inventory_sha256=inventory,exposed_weather=sorted(exposed),
        primary='Unchanged reserve20 versus V10 total loss on fresh buildings and weather windows.',
        secondary=['Request-pressure versus V10','Reserve20 versus request-pressure'],
        selection='No beneficial revision was identified in development PI/hold-reset tests; keep original reserve20 unchanged. First12structurallyvalid seeds; no reward exclusions.',
        scope='New generated buildings and previously undeclared nonoverlapping public weather windows at known stations; not unseen-climate/station generalization. Freshness audit covers locally recorded exposure only.',
        analysis=dict(cluster='building',clusters=12,draws=10000,seed=731,interval='percentile95',secondary_unadjusted=True),
        guardrails='Original mean/per-case/morning tolerances retained; report all failures and inactive allocations; no automatic notebook export.',
        resume='Completed episodes only; interrupted episodes restart from zero. Do not rerun generation after partial failure.')
    write(path,protocol);print('FRESH_POLICIES_LOCKED',sha(path));return protocol

def load_protocol(repo):
    protocol=read(repo/RUN/'protocol.json')
    if protocol.get('status')!='policies_locked_before_fresh_generation' or protocol.get('weather')!=list(WEATHER) or protocol.get('arms')!=list(ARMS) or protocol.get('total_jobs')!=72:raise ValueError('wrong frozen follow-up design')
    check_hashes(repo,{**protocol['preserved_frozen_sha256'],**protocol['additional_source_sha256']})
    return protocol

def validate_smoke_marker(marker,locked_hash,smoke_hash):
    if (marker.get('status')!='complete_verified_active_integration'
        or marker.get('locked_protocol_sha256')!=locked_hash
        or marker.get('smoke_protocol_sha256')!=smoke_hash
        or marker.get('jobs')!=3 or marker.get('steps')!=864
        or marker.get('allocation_checked_zone_steps',0)<=0
        or marker.get('reserve_zone_steps',0)<=0):
        raise ValueError('three-arm active integration smoke must pass before fresh generation')

def generate(repo):
    p=load_protocol(repo);root=repo/RUN;scenario=repo/SCENARIOS
    smoke_review=read(root/'smoke/verified_review.json')
    ph=sha(root/'protocol.json');sh=sha(root/'smoke/protocol.json')
    validate_smoke_marker(smoke_review,ph,sh)
    smoke_protocol=read(root/'smoke/protocol.json');check_hashes(repo,smoke_protocol['input_sha256'])
    smoke_tasks=[dict(building='building_9002',weather='singapore_2',policy=arm,steps=864) for arm in ARMS]
    if smoke_protocol['tasks']!=smoke_tasks or smoke_protocol['locked_protocol_sha256']!=ph:raise ValueError('smoke task grid differs')
    for task in smoke_tasks:valid_result(read(root/'smoke'/(task['policy']+'.result.json')),task,ph,sh,expected_steps=864)
    if (root/'cohort.json').exists():load_cohort(repo);print('FRESH_COHORT_ALREADY_VERIFIED');return
    if (root/'generation_started.json').exists() or scenario.exists():raise ValueError('partial generation exists; preserve logs and inspect; no automatic overwrite')
    if any((repo/'smart_buildings/buildings'/f'building_{s:04d}.yaml').exists() for s in range(SEED,SEED+120)):raise ValueError('seed range is no longer fresh')
    write(root/'generation_started.json',dict(protocol_sha256=sha(root/'protocol.json'),date=datetime.now(timezone.utc).isoformat()))
    from hvaclab import holdout
    # Absolute resolved target is checked before calling the generator, whose
    # internal cleanup is restricted to individual NEW seed subdirectories.
    if not scenario.resolve().is_relative_to((repo/'smart_buildings').resolve()):raise ValueError('unsafe generator target')
    buildings=holdout.build_valid_set(SEED,COUNT,list(WEATHER),SCENARIOS,max_attempts=120)
    tasks=expected_tasks(buildings);inputs={}
    for b in buildings:
        for w in WEATHER:
            folder=repo/SCENARIOS/b/w
            for name in ('config.gin','config.yaml','floor_plan.npy','zone_map.npy'):
                target=folder/name;inputs[target.relative_to(repo).as_posix()]=sha(target)
        target=repo/'smart_buildings/buildings'/(b+'.yaml');inputs[target.relative_to(repo).as_posix()]=sha(target)
    check_hashes(repo,p['preserved_frozen_sha256'])
    write(root/'cohort.json',dict(protocol_sha256=sha(root/'protocol.json'),buildings=buildings,tasks=tasks,input_sha256=inputs))
    print('FRESH_COHORT_GENERATED',len(tasks),'jobs')

def load_cohort(repo):
    p=load_protocol(repo);root=repo/RUN;c=read(root/'cohort.json')
    if c['protocol_sha256']!=sha(root/'protocol.json') or c['tasks']!=expected_tasks(c['buildings']):raise ValueError('cohort identity differs')
    check_hashes(repo,c['input_sha256']);return p,c

def valid_result(record,task,protocol_hash,cohort_hash,expected_steps=STEPS):
    if record.get('status')!='complete_verified' or record.get('task')!=task or record.get('protocol_sha256')!=protocol_hash or record.get('cohort_sha256')!=cohort_hash:raise ValueError('result identity/completion differs')
    s=record['sums'];n=task['steps']
    if n!=expected_steps or s['steps']!=expected_steps or any(not math.isfinite(s[k]) for k in ('reward','comfort','energy','carbon')):raise ValueError('short or nonfinite result')
    if any(s[k]<-1e-12 for k in ('comfort','energy','carbon')):raise ValueError('invalid penalty sign')
    if not math.isclose(-s['reward']/n,(s['comfort']+s['energy']+s['carbon'])/n,rel_tol=1e-8,abs_tol=1e-9):raise ValueError('reward decomposition differs')
    for key in ('loss','comfort','energy','carbon'):
        value=-s['reward']/n if key=='loss' else s[key]/n
        if not math.isclose(record['means'][key],value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('means differ')
    if task['policy']=='V10' and record['shadow_action_parity_steps']!=expected_steps:raise ValueError('V10 parity incomplete')
    if not record['morning_comfort'] or any(not 1<=m['steps']<=24 for m in record['morning_comfort'].values()):raise ValueError('invalid morning records')
    audit=record['allocation_audit']
    if any(not math.isfinite(audit[k]) or audit[k]<0 for k in ('max_relative_actual_flow_error','max_relative_cooling_error','max_supply_error_K')):raise ValueError('invalid physical audit values')
    if audit['max_relative_actual_flow_error']>2e-6 or audit['max_relative_cooling_error']>2e-6 or audit['max_supply_error_K']>.01:raise ValueError('physical allocation audit failed')
    return True

def run_job(repo,descriptor):
    import numpy as np
    from smart_buildings import evaluate as ev
    from hvaclab.pipeline_entry import register
    from hvaclab.registry import resolve_policy
    from hvaclab.zone_reserve_policy import ZoneReservePolicy
    from hvaclab.offhours_hold_worker import MorningAccumulator
    from request_pressure_baseline import RequestPressureBaseline
    root=repo/RUN;job=read(descriptor);task=job['task'];smoke=job.get('scope')=='integration_only'
    if smoke:
        p=load_protocol(repo);c=read(root/'smoke/protocol.json');ch=sha(root/'smoke/protocol.json');n=864
        if c['locked_protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('smoke lock differs')
        check_hashes(repo,c['input_sha256'])
    else:p,c=load_cohort(repo);ch=sha(root/'cohort.json');n=STEPS
    ph=sha(root/'protocol.json')
    if task not in c['tasks'] or job['protocol_sha256']!=ph or job['cohort_sha256']!=ch:raise ValueError('job identity differs')
    output=Path(job['result_path']).resolve()
    if not output.is_relative_to(root):raise ValueError('result outside run root')
    if output.exists():valid_result(read(output),task,ph,ch,expected_steps=n);return
    scenario_root=Path('smart_buildings/holdout_scenarios/dev') if smoke else SCENARIOS
    register();folder=repo/scenario_root/task['building']/task['weather'];env=ev.load_hybrid_environment(folder/'config.gin');meta=ev.create_building_meta_data(env,folder);spec=env.action_spec()
    if task['policy']=='V10':policy=ev.initialize_policy(resolve_policy('damper12_mid'),meta,spec)
    elif task['policy']=='reserve20':policy=ZoneReservePolicy(meta,spec,**p['v10_parameters'],**p['reserve_parameters'])
    else:policy=RequestPressureBaseline(meta,spec,**p['v10_parameters'],**p['request_parameters'])
    shadow=ev.initialize_policy(resolve_policy('damper12_mid'),meta,spec) if task['policy']=='V10' else None
    boxes={v.device_id():v for v in env.building.simulator.hvac.vavs.values()};supplies={}
    if task['policy']=='reserve20':
        for v,box in boxes.items():
            compute=box.compute_reheat_energy_rate
            def tap(supply,water,_compute=compute,_v=v):
                supplies.setdefault(_v,float(supply));return _compute(supply,water)
            box.compute_reheat_energy_rate=tap
    audit=dict(checked_zone_steps=0,max_relative_actual_flow_error=0.,max_relative_cooling_error=0.,max_supply_error_K=0.)
    morning=MorningAccumulator(meta['time_zone'],simulation_clock_naive=env.current_simulation_timestamp.tz is None)
    obs,_=env.reset();sums=dict(steps=0,reward=0.,comfort=0.,energy=0.,carbon=0.);parity=0;previous=None;started=time.monotonic()
    for i in range(n):
        action=ev.validate_action(policy.get_action(obs),spec)
        if shadow is not None:
            expected=ev.validate_action(shadow.get_action(obs),spec)
            if any(not np.array_equal(action[k],expected[k]) for k in action):raise ValueError('V10 shadow mismatch')
            parity+=1
        allocations=copy_allocations(policy) if task['policy']=='reserve20' else {};supplies.clear()
        obs,reward,end,truncated,_=env.step(action);info,response=env.get_reward_info_and_response()
        epoch=info.start_timestamp.seconds+info.start_timestamp.nanos/1e9
        if previous is not None and epoch!=previous+300:raise ValueError('non-contiguous scorer clock')
        previous=epoch;morning.add(epoch,response)
        for key,value in zip(('reward','comfort','energy','carbon'),(float(reward),-.6*float(response.normalized_productivity_regret),.2*float(response.normalized_energy_cost),.2*float(response.normalized_carbon_emission))):
            if not math.isfinite(value):raise ValueError('nonfinite reward')
            sums[key]+=value
        sums['steps']+=1
        for ahu,a in allocations.items():
            ctrl=next(x for x in policy.ahu_controllers if x['ahu_id']==ahu)
            for j,(v,_,_) in enumerate(ctrl['zones']):
                box=boxes[v]
                if abs(box._max_air_flow_static_pressure-policy.flow_design_pressure_pa)>1e-8 or abs(box._min_damper_setting-policy.damper_floor)>1e-8:raise ValueError('flow-law assumptions differ')
                actual=float(box.flow_rate_demand);requested=a['requested_flows'][j];t=a['planning_temperature'][j];m=a.get('target_cooling_multiplier',[1.]*len(a['old_flows']))[j]
                if not 1<=m<=1.2+1e-12:raise ValueError('undeclared reserve target')
                before=a['old_flows'][j]*(t-a['old_supply'])*m;after=actual*(t-a['new_supply'])
                if not np.all(np.isfinite([actual,requested,before,after,supplies.get(v,float('nan'))])) or requested<=0 or before<=0:raise ValueError('invalid physical allocation sample')
                audit['checked_zone_steps']+=1
                audit['max_relative_actual_flow_error']=max(audit['max_relative_actual_flow_error'],abs(actual/requested-1))
                audit['max_relative_cooling_error']=max(audit['max_relative_cooling_error'],abs(after/before-1))
                audit['max_supply_error_K']=max(audit['max_supply_error_K'],abs(supplies[v]-a['new_supply']))
        if (i+1)%288==0:print('HEARTBEAT',i+1,'/',n,flush=True)
        if (end or truncated) and i+1<n:raise ValueError('early termination')
    record=dict(status='complete_verified',task=task,protocol_sha256=ph,cohort_sha256=ch,sums=sums,
        means=dict(loss=-sums['reward']/n,comfort=sums['comfort']/n,energy=sums['energy']/n,carbon=sums['carbon']/n),morning_comfort=morning.days,
        allocation_audit=audit,shadow_action_parity_steps=parity,diagnostics=policy.hold_report() if hasattr(policy,'hold_report') else {},seconds=time.monotonic()-started,
        scope='Integration-only reused development scenario, not fresh cohort' if smoke else p['scope'])
    valid_result(record,task,ph,ch,expected_steps=n);load_protocol(repo);write(output,record);print('FRESH_JOB_VERIFIED',task,flush=True)

def copy_allocations(policy):
    import copy
    return copy.deepcopy(policy.last_allocations)

def smoke(repo):
    import psutil
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('smoke requires8GiB free on server')
    load_protocol(repo);root=repo/RUN;out=root/'smoke';out.mkdir(parents=True,exist_ok=True)
    folder=repo/'smart_buildings/holdout_scenarios/dev/building_9002/singapore_2'
    inputs={p.relative_to(repo).as_posix():sha(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.gin','.yaml','.npy')}
    if not (folder/'config.gin').exists():raise FileNotFoundError('integration scenario absent; install prior follow-up inputs first')
    csv=repo/'smart_buildings/weather/csv/singapore.csv';inputs[csv.relative_to(repo).as_posix()]=sha(csv)
    tasks=[dict(building='building_9002',weather='singapore_2',policy=arm,steps=864) for arm in ARMS]
    protocol=dict(scope='Integration only on reused development case; no fresh cohort outcomes',locked_protocol_sha256=sha(root/'protocol.json'),tasks=tasks,input_sha256=inputs)
    path=out/'protocol.json'
    if path.exists() and read(path)!=protocol:raise ValueError('prior smoke protocol differs; preserve it')
    if not path.exists():write(path,protocol)
    ph=sha(root/'protocol.json');ch=sha(path);results={}
    for task in tasks:
        name=task['policy'];result=out/(name+'.result.json');descriptor=out/(name+'.job.json')
        write(descriptor,dict(scope='integration_only',task=task,protocol_sha256=ph,cohort_sha256=ch,result_path=str(result)))
        if not result.exists():
            with (out/(name+'.log')).open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--repo',str(repo),'job','--descriptor',str(descriptor)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        d=read(result);valid_result(d,task,ph,ch,expected_steps=864);results[name]=d
    if results['reserve20']['allocation_audit']['checked_zone_steps']<=0:raise ValueError('smoke has no active reserve allocation; physical integration not exercised')
    if results['reserve20']['diagnostics'].get('reserve_zone_steps',0)<=0:raise ValueError('smoke has no selective reserve activation')
    write(out/'verified_review.json',dict(status='complete_verified_active_integration',locked_protocol_sha256=ph,smoke_protocol_sha256=ch,
        reused_case='building_9002/singapore_2',jobs=3,steps=864,allocation_checked_zone_steps=results['reserve20']['allocation_audit']['checked_zone_steps'],
        reserve_zone_steps=results['reserve20']['diagnostics']['reserve_zone_steps'],scope='Integration only, excluded from72fresh outcomes'))
    print('SMOKE_THREE_ARMS_AND_ACTIVE_PHYSICAL_AUDIT_VERIFIED')

def run(repo,workers,min_free):
    import psutil
    p,c=load_cohort(repo);root=repo/RUN;ph=sha(root/'protocol.json');ch=sha(root/'cohort.json');active=[];queue=[];complete=0
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('start fresh study on server with8GiB available')
    if min_free<2 or not math.isfinite(min_free):raise ValueError('worker memory gate must be finite and at least2GiB')
    if (root/'status.json').exists():
        for item in read(root/'status.json').get('active',[]):
            pid=item.get('pid')
            if pid and psutil.pid_exists(pid):
                try:command=' '.join(psutil.Process(pid).cmdline())
                except (psutil.AccessDenied,psutil.NoSuchProcess):raise RuntimeError('cannot prove prior worker is stopped; inspect status/PIDs before resume')
                if 'run_fresh_transfer.py' in command:raise RuntimeError('prior fresh worker still active; do not start a duplicate supervisor')
    for task in c['tasks']:
        key='__'.join(task[k] for k in ('building','weather','policy'));result=root/'jobs'/(key+'.result.json')
        if result.exists():valid_result(read(result),task,ph,ch);complete+=1;continue
        descriptor=result.with_suffix('.job.json');write(descriptor,dict(task=task,protocol_sha256=ph,cohort_sha256=ch,result_path=str(result)))
        queue.append((descriptor,result,task))
    try:
        while queue or active:
            for item in list(active):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();active.remove(item)
                if code!=0:raise RuntimeError('worker failed; inspect '+str(item['descriptor'].with_suffix('.log')))
                valid_result(read(item['result']),item['task'],ph,ch);complete+=1
            available=psutil.virtual_memory().available/2**30
            if queue and len(active)<workers and available>=min_free:
                descriptor,result,task=queue.pop(0);log=descriptor.with_suffix('.log').open('a')
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--repo',str(repo),'job','--descriptor',str(descriptor)],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
                active.append(dict(process=proc,log=log,descriptor=descriptor,result=result,task=task))
            write(root/'status.json',dict(status='running' if active else 'waiting_for_memory' if queue else 'complete_verified',completed=complete,total=72,available_gib=available,
                active=[dict(pid=x['process'].pid,task=x['task']) for x in active],updated_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=ph,cohort_sha256=ch))
            if queue or active:time.sleep(5)
    except BaseException as e:
        write(root/'status.json',dict(status='failed',completed=complete,total=72,error=repr(e),active=[dict(pid=x['process'].pid,task=x['task']) for x in active]));raise
    print('FRESH_72_JOBS_COMPLETE')

def morning_compare(base,candidate):
    if set(base)!=set(candidate):raise ValueError('morning dates differ')
    failed=[]
    for day,m in base.items():
        n=candidate[day]
        if m['steps']!=n['steps'] or m['first_occupied_epoch']!=n['first_occupied_epoch']:raise ValueError('morning windows differ')
        b=-.6*m['prod_regret']/m['steps'];a=-.6*n['prod_regret']/n['steps']
        if a>b+max(1e-6,.001*abs(b)):failed.append(dict(day=day,baseline=b,candidate=a,delta=a-b))
    return dict(days=len(base),failed_mornings=failed)

def review(repo):
    import numpy as np
    p,c=load_cohort(repo);root=repo/RUN;ph=sha(root/'protocol.json');ch=sha(root/'cohort.json');records={}
    for task in c['tasks']:
        key='__'.join(task[k] for k in ('building','weather','policy'));d=read(root/'jobs'/(key+'.result.json'));valid_result(d,task,ph,ch);records[task['building'],task['weather'],task['policy']]=d
    means={arm:{metric:float(np.mean([records[b,w,arm]['means'][metric] for b in c['buildings'] for w in WEATHER])) for metric in ('loss','comfort','energy','carbon')} for arm in ARMS}
    comparisons={}
    for candidate,reference in (('reserve20','V10'),('request_pressure','V10'),('reserve20','request_pressure')):
        base=np.array([np.mean([records[b,w,reference]['means']['loss'] for w in WEATHER]) for b in c['buildings']]);cand=np.array([np.mean([records[b,w,candidate]['means']['loss'] for w in WEATHER]) for b in c['buildings']])
        if np.any(base<=0):raise ValueError('undefined relative loss')
        indices=np.random.default_rng(731).integers(0,COUNT,size=(10000,COUNT));boot=100*(cand[indices].mean(axis=1)/base[indices].mean(axis=1)-1)
        cases=[]
        for b in c['buildings']:
            for w in WEATHER:
                x=records[b,w,reference];y=records[b,w,candidate]
                cases.append(dict(building=b,weather=w,reference=x['means'],candidate=y['means'],morning=morning_compare(x['morning_comfort'],y['morning_comfort'])))
        comparisons[candidate+'_vs_'+reference]=dict(primary=(candidate,reference)==('reserve20','V10'),mean_change_pct=float(100*(cand.mean()/base.mean()-1)),cluster_ci95=np.quantile(boot,[.025,.975]).tolist(),clusters=12,draws=10000,
            cases=cases,failed_mornings=sum(len(x['morning']['failed_mornings']) for x in cases),total_mornings=sum(x['morning']['days'] for x in cases),each_case_loss_within5pct=all(x['candidate']['loss']<=1.05*x['reference']['loss']+1e-10 for x in cases))
    write(root/'verified_review.json',dict(status='complete_verified',protocol_sha256=ph,cohort_sha256=ch,means=means,comparisons=comparisons,scope=p['scope'],notebook_exported=False,ready_to_submit=False))
    print('FRESH_TRANSFER_REVIEW_COMPLETE_VERIFIED')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','smoke','generate','review'):sub.add_parser(name)
    run_p=sub.add_parser('run');run_p.add_argument('--workers',type=int,choices=range(1,9),default=8);run_p.add_argument('--min-free-gib',type=float,default=8.)
    job=sub.add_parser('job');job.add_argument('--descriptor',type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve();os.chdir(repo);sys.path[:0]=[str(repo),str(Path(__file__).resolve().parent)]
    if args.command=='plan':lock_protocol(repo)
    elif args.command=='smoke':smoke(repo)
    elif args.command=='generate':generate(repo)
    elif args.command=='run':run(repo,args.workers,args.min_free_gib)
    elif args.command=='review':review(repo)
    else:run_job(repo,args.descriptor.resolve())
if __name__=='__main__':main()
