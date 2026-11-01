"""Independent, restartable supervision for the frozen gated-reheat study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from hvaclab.coverage import digest, write_json
from hvaclab.pi_calibration import RunLock

REPO=Path(__file__).resolve().parents[1]
RUN=REPO/'experiments/runs/gated_reheat_calibration'
MODULE='hvaclab.gated_reheat_calibration'


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError,ValueError):
        return {}


def study_processes():
    """Match this scheduler or workers with exact, resolved job-directory scope."""
    schedulers=[]
    workers=[]
    jobs=(RUN/'jobs').resolve()
    for proc in psutil.process_iter(['name','cmdline']):
        try:
            args=proc.info['cmdline'] or []
            if (proc.info['name'] or '').lower() not in ('python.exe','pythonw.exe','python','python3'):
                continue
            if '-m' in args and args[args.index('-m')+1:args.index('-m')+2]==[MODULE]:
                if Path(proc.cwd()).resolve()==REPO:
                    schedulers.append(proc)
            if '--job' in args:
                index=args.index('--job')+1
                if index<len(args):
                    job=Path(args[index])
                    if not job.is_absolute():job=Path(proc.cwd())/job
                    if job.resolve().parent==jobs:
                        workers.append(proc)
        except (psutil.NoSuchProcess,psutil.AccessDenied,OSError):
            continue
    return schedulers,workers


def frozen_changes():
    protocol=read_json(RUN/'protocol.json')
    if not protocol.get('source_sha256'):
        return ['missing/invalid protocol']
    expected=dict(protocol['source_sha256'])
    changes=[]
    for relative,sha in expected.items():
        path=REPO/relative
        if not path.is_file() or digest(path)!=sha:changes.append(relative)
    refs=protocol.get('reference_sha256',{})
    for split,sha in refs.items():
        path=REPO/f'experiments/runs/e_sync_coverage/{split}.json'
        if not path.is_file() or digest(path)!=sha:changes.append(str(path))
    curve=REPO/'experiments/runs/reheat_curve_calibration/rollouts.json'
    if not curve.is_file() or digest(curve)!=protocol.get('curve_reference_sha256'):
        changes.append(str(curve))
    return changes


class Supervisor:
    def __init__(self):
        self.child=None
        self.log=None
        self.next_launch=0.
        self.failures=0
        self.launches=0
        self.launched_at=None

    def state(self,status,**extra):
        write_json(RUN/'supervisor_status.json',dict(
            status=status,pid=psutil.Process().pid,
            updated_at=datetime.now(timezone.utc).isoformat(),
            launches=self.launches,consecutive_failures=self.failures,**extra))

    def event(self,kind,**extra):
        with (RUN/'supervisor_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                         event=kind,**extra))+'\n')

    def tick(self):
        if self.child is not None:
            code=self.child.poll()
            if code is not None:
                self.log.close()
                self.event('scheduler_exit',exit_code=code)
                self.child=self.log=None
                self.failures+=1
                self.next_launch=time.monotonic()+min(300,15*2**min(self.failures-1,5))
        schedulers,workers=study_processes()
        if (RUN/'supervisor.stop').exists():
            self.state('stopped',note='Supervision stopped; existing scheduler/workers left running')
            return False
        if schedulers or workers:
            # Adopt an already running scheduler. If it dies, let scoped orphan
            # workers finish before resuming, avoiding concurrent result writers.
            if schedulers and self.launched_at is not None and time.monotonic()-self.launched_at>300:
                self.failures=0
            self.state('monitoring' if schedulers else 'waiting_for_orphan_workers',
                       scheduler_pids=[p.pid for p in schedulers],worker_pids=[p.pid for p in workers])
            return True
        if read_json(RUN/'status.json').get('status')=='complete':
            self.state('complete')
            self.event('study_complete')
            return False
        changed=frozen_changes()
        if changed:
            self.state('blocked_inputs_changed',changed=changed)
            return True
        if self.failures>=5:
            self.state('blocked_repeated_failures',note='Five consecutive exits; inspect scheduler logs')
            return True
        if time.monotonic()<self.next_launch:
            self.state('restart_backoff')
            return True
        if psutil.virtual_memory().available<1.75*2**30:
            self.state('waiting_for_memory')
            return True
        import shutil
        if shutil.disk_usage(RUN).free<2**30:
            self.state('waiting_for_disk')
            return True
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path=RUN/f'scheduler_{stamp}.log'
        self.log=path.open('w',encoding='utf-8')
        try:
            self.child=subprocess.Popen([sys.executable,'-u','-m',MODULE],cwd=REPO,
                                        stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            self.log=None
            raise
        self.launches+=1
        self.launched_at=time.monotonic()
        self.event('scheduler_launch',child_pid=self.child.pid,log=str(path))
        self.state('launched',scheduler_pid=self.child.pid,log=str(path))
        return True


def main():
    global RUN,MODULE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=('gated_reheat_calibration','persistence_calibration'),
                        default='gated_reheat_calibration')
    args=parser.parse_args()
    RUN=REPO/'experiments/runs'/args.study
    MODULE='hvaclab.'+args.study
    RUN.mkdir(parents=True,exist_ok=True)
    with RunLock(RUN/'supervisor.lock'):
        supervisor=Supervisor()
        supervisor.event('supervisor_start')
        while supervisor.tick():time.sleep(15)


if __name__=='__main__':main()
