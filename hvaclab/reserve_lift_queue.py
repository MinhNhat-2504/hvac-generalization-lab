"""Wait for cutoff finalization and RAM, integrate once, then supervise reserve study."""
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock
from hvaclab import supervisor as monitor

ROOT=Path('experiments/runs/reserve_lift_short')


def main():
    with RunLock(ROOT/'supervisor.lock'):
        while True:
            try:prior=json.loads(Path('experiments/runs/cutoff_lift_short/finalization.json').read_text())
            except (OSError,ValueError):prior={}
            ram=psutil.virtual_memory().available/2**30
            ready=prior.get('status')=='complete' and ram>=1.75
            write_json(ROOT/'queue_status.json',dict(status='preflight' if ready else 'waiting_for_cutoff_and_memory',available_gib=ram,
                cutoff_finalized=prior.get('status')=='complete',updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            if ready:break
            time.sleep(15)
        p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
        for name,sha in p['source_sha256'].items():
            if digest(name)!=sha:raise ValueError('queued source changed: '+name)
        smoke=ROOT/'preflight';smoke.mkdir(exist_ok=True)
        task=dict(building='building_9002',weather='singapore_2',policy='reserve20',steps=864,scenarios_dir='smart_buildings/holdout_scenarios/dev')
        job=dict(task=task,fingerprint=fingerprint,source_sha256=p['source_sha256'],result_path=str(smoke/'result.json'),prior_trace=None)
        write_json(smoke/'job.json',job)
        if not (smoke/'verified.json').exists():
            with (smoke/'worker.log').open('w') as log:
                subprocess.run([sys.executable,'-u','-m','hvaclab.reserve_lift_worker','--job',str(smoke/'job.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
            from hvaclab.reserve_lift_verify import verify_reserve
            result=json.loads((smoke/'result.json').read_text());verify_reserve(result,job)
            if result['hold_diagnostics']['reserve_zone_steps']<=0:raise ValueError('integration reserve never activated')
            write_json(smoke/'verified.json',dict(protocol_sha256=fingerprint,result_sha256=digest(smoke/'result.json')))
        verified=json.loads((smoke/'verified.json').read_text())
        if verified['protocol_sha256']!=fingerprint or verified['result_sha256']!=digest(smoke/'result.json'):
            raise ValueError('stale integration result')
        monitor.RUN=ROOT.resolve();monitor.MODULE='hvaclab.reserve_lift_short'
        watcher=monitor.Supervisor()
        write_json(ROOT/'queue_status.json',dict(status='supervising',updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        while watcher.tick():time.sleep(15)


if __name__=='__main__':main()
