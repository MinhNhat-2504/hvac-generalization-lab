"""Sequential, dependency-aware queue for predeclared HVAC experiments."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import psutil

from hvaclab import supervisor as monitor
from hvaclab.coverage import digest
from hvaclab.pi_calibration import RunLock
from hvaclab.thermal_coast_entry import SPECS


def dependencies_ready(states,live_modules):
    return all(state.get('status')=='complete' and 'hvaclab.'+name not in live_modules
               for name,state in states.items())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=tuple(SPECS),required=True)
    study=parser.parse_args().study
    repo=Path(__file__).resolve().parents[1]
    out=repo/'experiments/runs'/study
    out.mkdir(parents=True,exist_ok=True)
    dependencies=('persistence_calibration',)+SPECS[study]['dependencies']
    monitor.RUN=out
    monitor.MODULE='hvaclab.'+study
    with RunLock(out/'supervisor.lock'):
        watcher=monitor.Supervisor()
        watcher.event('queue_start',dependencies=dependencies)
        while True:
            if (out/'supervisor.stop').exists():
                watcher.state('stopped')
                return
            states={name:monitor.read_json(repo/'experiments/runs'/name/'status.json')
                    for name in dependencies}
            live=set()
            for proc in psutil.process_iter(['cmdline']):
                try:live.update(a for a in (proc.info['cmdline'] or []) if a.startswith('hvaclab.'))
                except (psutil.NoSuchProcess,psutil.AccessDenied):continue
            if dependencies_ready(states,live):break
            watcher.state('waiting_for_preceding_study',
                          dependencies={name:dict(status=state.get('status','not_started'),
                                                 completed=state.get('completed',0))
                                        for name,state in states.items()})
            time.sleep(15)
        with (out/'queue_prepare.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m',monitor.MODULE,'--prepare-only'],
                                  cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            watcher.state('blocked_preparation_failed',exit_code=result.returncode)
            return
        original=monitor.frozen_changes
        def inputs():
            changes=original()
            protocol=monitor.read_json(out/'protocol.json')
            for name,sha in protocol['comparison_sha256'].items():
                path=repo/name
                if not path.is_file() or digest(path)!=sha:changes.append(name)
            return changes
        monitor.frozen_changes=inputs
        watcher.event('dependencies_complete_start_next')
        while watcher.tick():time.sleep(15)


if __name__=='__main__':main()
