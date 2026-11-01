"""Wait for persistence validation, then supervise the prepared release study."""
import json
from pathlib import Path
import subprocess
import sys
import time

from hvaclab import supervisor as monitor
from hvaclab.coverage import digest
from hvaclab.pi_calibration import RunLock


def ready(state,schedulers_alive):
    return state.get('status')=='complete' and not schedulers_alive


def main():
    repo=Path(__file__).resolve().parents[1]
    previous=repo/'experiments/runs/persistence_calibration'
    out=repo/'experiments/runs/hold_release_calibration'
    out.mkdir(parents=True,exist_ok=True)
    # Use the same independent supervisor, with a scoped directory and module.
    monitor.RUN=out
    monitor.MODULE='hvaclab.hold_release_calibration'
    with RunLock(out/'supervisor.lock'):
        watcher=monitor.Supervisor()
        watcher.event('queue_start',dependency=str(previous))
        while True:
            if (out/'supervisor.stop').exists():
                watcher.state('stopped')
                return
            state=monitor.read_json(previous/'status.json')
            import psutil
            alive=False
            for proc in psutil.process_iter(['cmdline']):
                try:
                    if 'hvaclab.persistence_calibration' in (proc.info['cmdline'] or []):alive=True
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    continue
            if ready(state,alive):break
            watcher.state('waiting_for_preceding_study',dependency_status=state.get('status'),
                          dependency_completed=state.get('completed'))
            time.sleep(15)
        # The source/protocol snapshot is created only after prior validation
        # finishes, so it can include a valid selected winner as comparator.
        with (out/'queue_prepare.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m',monitor.MODULE,'--prepare-only'],
                                  cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            watcher.state('blocked_preparation_failed',exit_code=result.returncode)
            return
        original=monitor.frozen_changes
        def inputs():
            changes=original()
            p=monitor.read_json(out/'protocol.json')
            for filename,key in (('status.json','preceding_status_sha256'),
                                 ('rollouts.json','preceding_results_sha256')):
                path=previous/filename
                if digest(path)!=p[key]:changes.append(str(path))
            return changes
        monitor.frozen_changes=inputs
        watcher.event('dependency_complete_start_next')
        while watcher.tick():time.sleep(15)


if __name__=='__main__':main()
