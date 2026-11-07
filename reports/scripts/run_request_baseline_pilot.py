"""Resumable exploratory comparison on two reused development cases.

Each job runs in a fresh process with a fresh environment and controller.
This script does not modify the frozen 192-run cohort or export a notebook.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ARMS=('V10','reserve20','request_pressure')
CASES=(('building_9001','atlanta_4'),('building_9002','singapore_2'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(data):return (json.dumps(data,sort_keys=True,indent=2)+'\n').encode()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(canonical(data));temporary.replace(path)
def tasks(steps):
    if steps not in (864,4032):raise ValueError('pilot only: choose864smoke or4032prefix')
    return [dict(building=b,weather=w,policy=a,steps=steps) for b,w in CASES for a in ARMS]
def valid_result(record,task,fingerprint):
    if record.get('status')!='complete_verified' or record.get('task')!=task or record.get('protocol_sha256')!=fingerprint:
        raise ValueError('wrong result identity or incomplete job')
    s=record['sums'];n=task['steps']
    if s['steps']!=n or any(not math.isfinite(s[k]) for k in ('reward','comfort','energy','carbon')):
        raise ValueError('short/nonfinite result')
    if any(s[k]<-1e-12 for k in ('comfort','energy','carbon')):raise ValueError('invalid penalty sign')
    if not math.isclose(-s['reward']/n,(s['comfort']+s['energy']+s['carbon'])/n,rel_tol=1e-8,abs_tol=1e-9):
        raise ValueError('reward decomposition mismatch')
    for key in ('loss','comfort','energy','carbon'):
        expected=-s['reward']/n if key=='loss' else s[key]/n
        if not math.isclose(record['means'][key],expected,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('mean mismatch')
    for m in record['morning_comfort'].values():
        if not 1<=m['steps']<=24:raise ValueError('invalid morning window')
    return True


def create_protocol(repo,steps):
    original=repo/'experiments/runs/paper_confirmation_20261106/protocol.json'
    if not original.is_file():raise FileNotFoundError('retain the original confirmation protocol at '+str(original))
    frozen=json.loads(original.read_text())['source_sha256']
    for n,h in frozen.items():
        if digest(repo/n)!=h:raise ValueError('frozen source/input changed: '+n)
    source=Path(__file__).resolve();baseline=source.with_name('request_pressure_baseline.py')
    inputs={}
    for b,w in CASES:
        folder=repo/'smart_buildings/holdout_scenarios/dev'/b/w
        if not (folder/'config.gin').is_file():raise FileNotFoundError('missing pilot scenario '+str(folder))
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.gin','.yaml','.yml','.npy','.csv'):
                inputs[p.relative_to(repo).as_posix()]=digest(p)
        csv=repo/'smart_buildings/weather/csv'/(w.rsplit('_',1)[0]+'.csv')
        if csv.exists():inputs[csv.relative_to(repo).as_posix()]=digest(csv)
    return dict(scope='Exploratory comparison after the192-run confirmation, on reused development buildings/weather; not independent validation or full challenge episodes.',
        tasks=tasks(steps),steps=steps,original_server_protocol_sha256=digest(original),
        preserved_frozen_sha256=frozen,scenario_input_sha256=inputs,
        pilot_script_sha256=digest(source),baseline_script_sha256=digest(baseline),
        baseline_settings=dict(request_reset=True,pressure_floor_pa=200.,trim_pa=10.,respond_pa=15.,max_response_pa=37.,
            request_on=.95,request_off=.85,ignored_requests=0,update_steps=1,startup_delay_steps=1),
        reference='Resolved damper12_mid V10 parameters; reserve20 unchanged, projected cooling allocation and beta0.2.',
        reset='Fresh process, environment and policy per job. Completed-job cache only; interrupted jobs restart from zero.',
        safety='Pressure-only request baseline changes airflow and may worsen comfort; no G36 compliance or safety claim.',
        notebook_exported=False)


def run_job(repo,descriptor):
    import numpy as np
    from smart_buildings import evaluate as ev
    from hvaclab.pipeline_entry import register
    from hvaclab.registry import _VARIANTS,resolve_policy
    from hvaclab.zone_reserve_policy import ZoneReservePolicy
    from hvaclab.offhours_hold_worker import MorningAccumulator
    from request_pressure_baseline import RequestPressureBaseline
    job=json.loads(descriptor.read_text());task=job['task'];protocol_path=Path(job['protocol_path'])
    protocol=json.loads(protocol_path.read_text());fingerprint=digest(protocol_path)
    if fingerprint!=job['protocol_sha256'] or task not in protocol['tasks']:raise ValueError('job protocol mismatch')
    if digest(__file__)!=protocol['pilot_script_sha256'] or digest(Path(__file__).with_name('request_pressure_baseline.py'))!=protocol['baseline_script_sha256']:
        raise ValueError('pilot script changed')
    for n,h in {**protocol['preserved_frozen_sha256'],**protocol['scenario_input_sha256']}.items():
        if digest(repo/n)!=h:raise ValueError('input changed '+n)
    output=Path(job['result_path'])
    if output.exists():
        valid_result(json.loads(output.read_text()),task,fingerprint);return
    register();folder=repo/'smart_buildings/holdout_scenarios/dev'/task['building']/task['weather']
    env=ev.load_hybrid_environment(folder/'config.gin');meta=ev.create_building_meta_data(env,folder);spec=env.action_spec()
    settings=dict(_VARIANTS['damper12_mid'][1])
    if task['policy']=='V10':policy=ev.initialize_policy(resolve_policy('damper12_mid'),meta,spec)
    elif task['policy']=='reserve20':
        policy=ZoneReservePolicy(meta,spec,**settings,flow_equivalent=True,cooling_lift=True,project_lift=True,comfort_guard=False,reserve_fraction=.2)
    elif task['policy']=='request_pressure':policy=RequestPressureBaseline(meta,spec,**settings,**protocol['baseline_settings'])
    else:raise ValueError('unknown pilot policy')
    morning=MorningAccumulator(meta['time_zone'],simulation_clock_naive=env.current_simulation_timestamp.tz is None)
    obs,_=env.reset();sums=dict(steps=0,reward=0.,comfort=0.,energy=0.,carbon=0.,occupied_steps=0)
    start=time.monotonic()
    for i in range(task['steps']):
        action=ev.validate_action(policy.get_action(obs),spec)
        obs,reward,end,truncated,_=env.step(action)
        info,response=env.get_reward_info_and_response()
        values=(float(reward),-.6*float(response.normalized_productivity_regret),.2*float(response.normalized_energy_cost),.2*float(response.normalized_carbon_emission))
        if not np.all(np.isfinite(values)):raise ValueError('nonfinite score')
        for key,value in zip(('reward','comfort','energy','carbon'),values):sums[key]+=value
        sums['steps']+=1;sums['occupied_steps']+=int(response.total_occupancy>0)
        epoch=info.start_timestamp.seconds+info.start_timestamp.nanos/1e9;morning.add(epoch,response)
        if (i+1)%288==0:print('HEARTBEAT',i+1,'/',task['steps'],flush=True)
        if (end or truncated) and i+1<task['steps']:raise ValueError('unexpected short pilot episode')
    n=task['steps'];record=dict(status='complete_verified',task=task,protocol_sha256=fingerprint,sums=sums,
        means=dict(loss=-sums['reward']/n,comfort=sums['comfort']/n,energy=sums['energy']/n,carbon=sums['carbon']/n),
        morning_comfort=morning.days,clock_semantics=dict(simulation_clock_naive=morning.simulation_clock_naive),
        controller_diagnostics=policy.hold_report() if hasattr(policy,'hold_report') else {},seconds=time.monotonic()-start,
        scope=protocol['scope'],notebook_exported=False)
    valid_result(record,task,fingerprint);write(output,record);print('PILOT_JOB_VERIFIED',task,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,default=Path('experiments/runs/request_baseline_pilot'))
    parser.add_argument('--steps',type=int,choices=(864,4032),default=4032)
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--job',type=Path)
    args=parser.parse_args();repo=args.repo.resolve();os.chdir(repo);sys.path[:0]=[str(repo),str(Path(__file__).resolve().parent)]
    if args.job:run_job(repo,args.job.resolve());return
    import psutil
    root=args.output.resolve()
    if not root.is_relative_to(repo/'experiments/runs'):raise ValueError('pilot output must stay under experiments/runs')
    root.mkdir(parents=True,exist_ok=True);protocol=create_protocol(repo,args.steps);pp=root/'protocol.json'
    if pp.exists() and pp.read_bytes()!=canonical(protocol):raise ValueError('protocol differs; choose a fresh output directory')
    if not pp.exists():write(pp,protocol)
    fingerprint=digest(pp);queue=[];complete=0;active=[]
    for task in protocol['tasks']:
        key='__'.join(task[k] for k in ('building','weather','policy'));result=root/(key+'.result.json')
        if result.exists():valid_result(json.loads(result.read_text()),task,fingerprint);complete+=1;continue
        descriptor=root/(key+'.job.json');write(descriptor,dict(task=task,protocol_path=str(pp),protocol_sha256=fingerprint,result_path=str(result)))
        queue.append((descriptor,result,task))
    try:
        while queue or active:
            for item in list(active):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();active.remove(item)
                if code!=0:raise RuntimeError('pilot worker failed; inspect '+str(item['descriptor'].with_suffix('.log')))
                valid_result(json.loads(item['result'].read_text()),item['task'],fingerprint);complete+=1
            available=psutil.virtual_memory().available/2**30
            if queue and len(active)<args.workers and available>=1.75:
                descriptor,result,task=queue.pop(0);log=descriptor.with_suffix('.log').open('a')
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--repo',str(repo),'--job',str(descriptor)],stdout=log,stderr=subprocess.STDOUT)
                active.append(dict(process=process,log=log,descriptor=descriptor,result=result,task=task))
            status='running' if active else 'waiting_for_memory' if queue else 'complete_verified'
            write(root/'status.json',dict(status=status,completed=complete,total=6,
                active=[dict(pid=i['process'].pid,task=i['task']) for i in active],available_gib=available,minimum_start_gib=1.75,
                updated_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=fingerprint))
            if queue or active:time.sleep(3)
        print('PILOT_SIX_JOBS_COMPLETE_VERIFIED',flush=True)
    except BaseException as error:
        write(root/'status.json',dict(status='failed',completed=complete,total=6,error=repr(error),
            active=[dict(pid=i['process'].pid,task=i['task']) for i in active],protocol_sha256=fingerprint))
        # Do not terminate valid active jobs; their completed results can resume later.
        raise


if __name__=='__main__':main()
