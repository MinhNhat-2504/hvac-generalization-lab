"""Independent supervisor for the fixed short phase ablation."""
from pathlib import Path
import subprocess
import sys
import time
from hvaclab import supervisor as monitor
from hvaclab.pi_calibration import RunLock


def main():
    repo=Path(__file__).resolve().parents[1];run=repo/'experiments/runs/comfort_band_short';run.mkdir(parents=True,exist_ok=True)
    monitor.RUN=run;monitor.MODULE='hvaclab.comfort_band_short'
    with RunLock(run/'supervisor.lock'):
        if not (run/'protocol.json').exists():
            with (run/'prepare.log').open('w') as log:
                subprocess.run([sys.executable,'-m',monitor.MODULE,'--prepare-only'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        watcher=monitor.Supervisor()
        while watcher.tick():time.sleep(15)


if __name__=='__main__':main()
