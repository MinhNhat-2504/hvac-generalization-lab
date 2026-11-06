"""Prepare and execute a fixed fresh-building confirmation cohort on a server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/paper_confirmation_20261106')
SCENARIOS=Path('smart_buildings/holdout_paper_confirmation_20261106')
ARMS=('phase_ctrl','flow_only','project_lift','reserve20')
WEATHER=('fairbanks_1','phoenix_5','atlanta_4','singapore_2')
SEED_START=2026110700


def tasks(buildings):
    return [dict(building=b,weather=w,policy=a,steps=8640,scenarios_dir=SCENARIOS.as_posix())
            for b in buildings for w in WEATHER for a in ARMS]


def plan():
    source=list(Path('hvaclab').rglob('*.py'))+list(Path('smart_buildings').rglob('*.py'))
    source+=list(Path('smart_buildings/weather').glob('*.yaml'))
    source+=list(Path('smart_buildings/weather/csv').glob('*.csv'))
    source+=[Path('pyproject.toml')]
    payload=dict(steps=8640,building_count=12,seed_start=SEED_START,max_seed_attempts=120,
        arms=ARMS,weather=WEATHER,source_sha256={p.as_posix():digest(p) for p in sorted(set(source))},
        scope='New generated building seeds, conditioned on four previously used weather strips. Not unseen-climate or organizer evaluation.',
        selection='First12 structurally valid generated buildings,28-85zones. No reward-based exclusions, no parameter changes after outcomes.',
        analysis='Equal-weight scenarios; paired differences; bootstrap by building(seed731,10000draws); report all cases, daily morning failures, energy/comfort/carbon. Deterministic repeats are not independent samples.',
        primary='reserve20 versusV10 total loss; report paired cluster95pctCI and comfort/morning regressions regardless of significance. Packaging gates unchanged; no automatic promotion.',
        robustness='No new robustness or causal intervention is included in this confirmation batch.')
    payload=json.loads(json.dumps(payload));path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=payload:raise ValueError('frozen inputs changed; preserve protocol')
    write_json(path,payload);return payload


def check(p):
    for name,sha in p['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('frozen source/input changed: '+name)


def generate(p):
    from hvaclab import holdout
    # Exact workspace-scoped root; the existing generator only retries/removes
    # candidate-building directories beneath this new cohort root.
    repo=Path.cwd().resolve();out=SCENARIOS.resolve()
    if not out.is_relative_to(repo) or out==repo:raise ValueError('unsafe cohort root')
    if SCENARIOS.exists() and any(SCENARIOS.iterdir()):raise ValueError('cohort exists; do not overwrite')
    for seed in range(SEED_START,SEED_START+p['max_seed_attempts']):
        if (holdout.BUILDINGS_DIR/f'building_{seed:04d}.yaml').exists():raise ValueError('fresh seed already materialized')
    SCENARIOS.mkdir(parents=True,exist_ok=True)
    buildings=holdout.build_valid_set(SEED_START,12,list(WEATHER),SCENARIOS,max_attempts=120)
    if len(buildings)!=12:raise ValueError('insufficient structurally valid buildings; preserve generation logs')
    inputs={f.as_posix():digest(f) for f in SCENARIOS.rglob('*') if f.is_file()}
    write_json(ROOT/'cohort.json',dict(buildings=buildings,tasks=tasks(buildings),input_sha256=inputs))


def run(p,workers,min_gib):
    from hvaclab.reserve_lift_verify import verify_reserve
    cohort=json.loads((ROOT/'cohort.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for path,sha in cohort['input_sha256'].items():
        if digest(path)!=sha:raise ValueError('cohort input changed')
    pending=[];active=[];done=[]
    for task in cohort['tasks']:
        stem='__'.join(task[k] for k in ('building','weather','policy'))
        path=ROOT/'jobs'/(stem+'.job.json');result=ROOT/'jobs'/(stem+'.result.json')
        job=dict(task=task,fingerprint=fingerprint,source_sha256=p['source_sha256'],result_path=str(result),prior_trace=None)
        write_json(path,job)
        if result.exists() and result.with_suffix('.thermal.json').exists():
            r=json.loads(result.read_text());verify_reserve(r,job);done.append(r)
        else:pending.append((path,result,job))
    try:
        while active or pending:
            for proc,log,result,job in list(active):
                if proc.poll() is None:continue
                log.close();active.remove((proc,log,result,job))
                if proc.returncode:raise RuntimeError('worker failed; inspect '+str(result))
                r=json.loads(result.read_text());verify_reserve(r,job);done.append(r)
            free=psutil.virtual_memory().available/2**30
            if pending and len(active)<workers and free>=min_gib:
                path,result,job=pending.pop(0);log=path.with_suffix('.log').open('w')
                proc=subprocess.Popen([sys.executable,'-u','-m','hvaclab.paper_validation_worker','--job',str(path)],stdout=log,stderr=subprocess.STDOUT)
                active.append((proc,log,result,job))
            write_json(ROOT/'status.json',dict(status='running' if active or pending else 'complete',completed=len(done),total=len(cohort['tasks']),active=[j['task'] for _,_,_,j in active],available_gib=free,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            write_json(ROOT/'rollouts.json',dict(records=done));time.sleep(5)
    finally:
        for proc,log,_,_ in active:
            if proc.poll() is None:
                try:
                    parent=psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):child.terminate()
                    parent.terminate()
                except psutil.NoSuchProcess:pass
            log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','generate','run'))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--min-free-gib',type=float,default=2.)
    args=parser.parse_args()
    if args.workers<1 or args.min_free_gib<1:raise ValueError('invalid resource limits')
    ROOT.mkdir(parents=True,exist_ok=True)
    with RunLock(ROOT/'scheduler.lock'):
        p=plan();check(p)
        if args.action=='generate':generate(p)
        elif args.action=='run':run(p,args.workers,args.min_free_gib)
        print('PAPER_VALIDATION',args.action,'complete')


if __name__=='__main__':main()
