"""Resumable one-worker short transfer/feedback screen on buildings9001 and9002."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import write_json,digest,valid_result
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/online_forecast_short')
PAIRS=(('building_9001','denver_1'),('building_9002','atlanta_4'))


def prepare():
    files=list(Path('hvaclab/policies').glob('*.py'))+list(Path('smart_buildings').rglob('*.py'))
    files += [Path(p) for p in ('hvaclab/online_forecast_short.py','hvaclab/online_forecast_worker.py',
        'hvaclab/online_forecast_queue.py','hvaclab/runner.py','hvaclab/registry.py','hvaclab/pipeline_entry.py',
        'hvaclab/coverage.py','hvaclab/pi_calibration.py','hvaclab/supervisor.py')]
    for building,weather in PAIRS:files.extend((Path('smart_buildings/holdout_scenarios/dev')/building/weather).glob('*'))
    protocol=dict(steps=2016,pairs=PAIRS,policies=['online_shadow','online_quarter'],workers=1,
        warmup_steps=288,minimum_zone_samples=24,minimum_group_frames=24,blend=.25,correction_limit_k=.5,
        available_memory_gib_per_launch=1.75,
        gate='shadow matches every V10 action; online selected MAE below aged overall and occupied no worse in each building before feedback; all feedback arms active; mean reward loss at least1% below V10 and mean comfort no worse; otherwise stop short',
        scope='two non9000 buildings, selected previously used dev cases; short screen only, not independent hidden validation',
        source_sha256={p.as_posix():digest(p) for p in sorted(set(files)) if p.is_file()},
        curve_reference_sha256=digest('experiments/runs/reheat_curve_calibration/rollouts.json'))
    protocol=json.loads(json.dumps(protocol));path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('frozen protocol changed')
    write_json(path,protocol)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with RunLock(ROOT/'scheduler.lock'):
        protocol=prepare();fingerprint=digest(ROOT/'protocol.json');records=[]
        tasks=[dict(building=b,weather=w,policy=p,steps=protocol['steps'],scenarios_dir='smart_buildings/holdout_scenarios/dev') for p in protocol['policies'] for b,w in PAIRS]
        def checkpoint(state,active=None,waiting=False,decision=None):
            write_json(ROOT/'status.json',dict(status=state,completed=len(records),total=len(tasks),active=active,
                       waiting_for_memory=waiting,failed=[],decision=decision or {},updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            write_json(ROOT/'rollouts.json',dict(records=records))
        checkpoint('prepared')
        if args.prepare_only:return
        for task in tasks:
            if task['policy']=='online_quarter':
                shadows=[r for r in records if r['task']['policy']=='online_shadow']
                qualified=True
                for r in shadows:
                    m=r['online_diagnostics']['prequential_mae'];counts=r['online_diagnostics']['scored_zone_steps']
                    if not m['occupied'] or not m['empty'] or m['occupied'][3]>m['occupied'][0]+1e-10:
                        qualified=False;continue
                    selected=sum(m[g][3]*counts[i] for i,g in enumerate(('empty','occupied')))
                    aged=sum(m[g][0]*counts[i] for i,g in enumerate(('empty','occupied')))
                    if selected>=aged:qualified=False
                if not qualified:
                    checkpoint('complete',decision=dict(passes_shadow_forecast_gate=False,passes_short_gate=False,
                        stop_reason='causal shadow forecast did not beat aged sensor in both buildings; no feedback rollout',
                        full_sweep_started=False,notebook_exported=False));return
            stem='__'.join(task[k] for k in ('building','weather','policy'));result=ROOT/'jobs'/(stem+'.result.json');job=ROOT/'jobs'/(stem+'.job.json')
            write_json(job,dict(task=task,fingerprint=fingerprint,source_sha256=protocol['source_sha256'],result_path=str(result)))
            if result.exists():
                r=json.loads(result.read_text())
                if r['fingerprint']!=fingerprint or r['task']!=task or not valid_result(r['result'],task):raise ValueError('invalid cached rollout')
            else:
                while psutil.virtual_memory().available<1.75*2**30:
                    checkpoint('running',waiting=True);time.sleep(15)
                checkpoint('running',active=task)
                with job.with_suffix('.log').open('w') as log:
                    process=subprocess.run([sys.executable,'-u','-m','hvaclab.online_forecast_worker','--job',str(job)],stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:checkpoint('failed',decision=dict(error='worker exited',code=process.returncode));raise RuntimeError('worker failed; see '+str(job))
                r=json.loads(result.read_text())
                if not valid_result(r['result'],task):raise ValueError('invalid worker result')
            if task['policy']=='online_shadow' and r['shadow_action_parity_steps']!=task['steps']:raise ValueError('shadow parity incomplete')
            records.append(r);checkpoint('running')
        baseline=[r for r in records if r['task']['policy']=='online_shadow'];candidate=[r for r in records if r['task']['policy']=='online_quarter']
        def mean(rows,field,weight=1.):return sum(abs(r['result'][field]/r['result']['steps'])*weight for r in rows)/len(rows)
        control=mean(baseline,'total_reward');score=mean(candidate,'total_reward');comfort=mean(candidate,'prod_regret',.6);base_comfort=mean(baseline,'prod_regret',.6)
        active=all(r['online_diagnostics']['correction_zone_steps']>0 for r in candidate)
        decision=dict(v10_short_score=control,candidate_short_score=score,change_pct=100*(score/control-1),
            v10_comfort=base_comfort,candidate_comfort=comfort,correction_active_in_each_case=active,
            passes_short_gate=bool(active and score<.99*control and comfort<=base_comfort),
            full_sweep_started=False,notebook_exported=False)
        checkpoint('complete',decision=decision);print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
