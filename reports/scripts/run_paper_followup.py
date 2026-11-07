"""Run independent diagnostic processes on Fitlab; no fresh cohort generation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parallel-baseline',action='store_true')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    import psutil
    if psutil.virtual_memory().available<8*2**30:raise RuntimeError('Fitlab launcher requires8GiB available; do not use on low-memory workstation')
    root=repo/'experiments/runs/paper_followup_20261107';root.mkdir(parents=True,exist_ok=True)
    plan=repo/'reports/paper_submission/generated/memory_study/branch_plan.json'
    scripts=repo/'reports/scripts';processes=[]
    def launch(label,command):
        log=(root/(label+'.log')).open('a')
        proc=subprocess.Popen([sys.executable,'-u',*map(str,command)],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        processes.append(dict(label=label,process=proc,log=log))
    for case in ('singapore','atlanta'):
        result=root/('memory_'+case)/'result.json'
        if result.exists():
            if json.loads(result.read_text()).get('status')!='matched_state_factorial_verified':raise ValueError('prior memory run failed; preserve and inspect '+str(result))
            continue
        launch('memory_'+case,[scripts/'isolate_controller_memory.py','--case',case,'--plan',plan,'--output',root/('memory_'+case)])
    baseline_started=False
    def baseline():
        nonlocal baseline_started
        launch('baseline',[scripts/'run_request_baseline_pilot.py','--repo',repo,'--output',root/'baseline','--steps','4032','--workers','2'])
        baseline_started=True
    if args.parallel_baseline:baseline()
    while processes or not baseline_started:
        for item in list(processes):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();processes.remove(item)
            if code!=0:raise RuntimeError(item['label']+' failed; inspect its log; other independent jobs are not terminated')
            if item['label'].startswith('memory_'):
                result=root/item['label']/'result.json'
                if not result.exists() or json.loads(result.read_text()).get('status')!='matched_state_factorial_verified':raise ValueError('memory job did not produce verified result')
        if not processes and not baseline_started:baseline()
        state=dict(status='running' if processes else 'complete',updated_at=datetime.now(timezone.utc).isoformat(),
            active=[dict(label=i['label'],pid=i['process'].pid) for i in processes],baseline_started=baseline_started,
            fresh_validation_launched=False,scope='Development diagnostics/comparison only, not added to original192 confirmation.')
        tmp=root/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'status.json')
        if processes:time.sleep(5)
    print('FOLLOWUP_MEMORY_AND_BASELINE_COMPLETE; fresh cohort remains locked',flush=True)


if __name__=='__main__':main()
