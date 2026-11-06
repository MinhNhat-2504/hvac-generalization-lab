"""CPU/RAM pilot and physical integration before the independent paper cohort."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import digest,write_json
from hvaclab.paper_validation import ROOT,check
from hvaclab.reserve_lift_verify import verify_reserve
from hvaclab.pi_calibration import RunLock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--copies',type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or not 1<=args.copies<=8:raise ValueError('pilot workers/copies must be1-8')
    protocol=json.loads((ROOT/'protocol.json').read_text());check(protocol)
    folder=ROOT/f'preflight_{args.workers}workers_{args.copies}copies';folder.mkdir(parents=True,exist_ok=True)
    with RunLock(folder/'pilot.lock'):
        if (folder/'verified.json').exists():raise ValueError('pilot already complete; read its result')
        pending=[];active=[];records=[];peak=0;started=time.monotonic()
        for copy in range(args.copies):
            for arm,b,w,scenarios in (('phase_ctrl','building_train_1','fairbanks_1','smart_buildings/training_scenarios'),
                    ('reserve20','building_9002','singapore_2','smart_buildings/holdout_scenarios/dev')):
                task=dict(building=b,weather=w,policy=arm,steps=864,scenarios_dir=scenarios)
                stem=f'{copy}_{arm}';result=folder/(stem+'.result.json');path=folder/(stem+'.job.json')
                job=dict(task=task,fingerprint=digest(ROOT/'protocol.json'),source_sha256=protocol['source_sha256'],result_path=str(result),prior_trace=None)
                write_json(path,job);pending.append((path,result,job))
        try:
            while pending or active:
                for item in list(active):
                    proc,log,result,job=item
                    if proc.poll() is None:continue
                    log.close();active.remove(item)
                    if proc.returncode:raise RuntimeError('pilot worker failed: '+str(result))
                    r=json.loads(result.read_text());verify_reserve(r,job)
                    if job['task']['policy']=='reserve20' and r['hold_diagnostics']['reserve_zone_steps']<=0:
                        raise ValueError('reserve pilot never activated')
                    records.append(r)
                if pending and len(active)<args.workers and psutil.virtual_memory().available>=2*2**30:
                    path,result,job=pending.pop(0);log=path.with_suffix('.log').open('w')
                    proc=subprocess.Popen([sys.executable,'-u','-m','hvaclab.paper_validation_worker','--job',str(path)],stdout=log,stderr=subprocess.STDOUT)
                    active.append((proc,log,result,job))
                resident=0
                for proc,_,_,_ in active:
                    try:
                        parent=psutil.Process(proc.pid)
                        resident+=sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                peak=max(peak,resident)
                write_json(folder/'status.json',dict(completed=len(records),total=2*args.copies,active=len(active),peak_summed_rss_gib=peak/2**30))
                time.sleep(1)
        finally:
            for proc,log,_,_ in active:
                try:
                    parent=psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):child.terminate()
                    parent.terminate()
                except psutil.NoSuchProcess:pass
                log.close()
        result=dict(status='complete_verified',workers=args.workers,copies=args.copies,
            seconds=time.monotonic()-started,peak_summed_rss_gib=peak/2**30,
            note='Repeated deterministic cases benchmark resource use only; excluded from confirmation statistics.',
            protocol_sha256=digest(ROOT/'protocol.json'),python=sys.version,
            completed=len(records),available_gib=psutil.virtual_memory().available/2**30)
        write_json(folder/'verified.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
