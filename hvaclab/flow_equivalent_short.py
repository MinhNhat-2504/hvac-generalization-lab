"""Flow-equivalent pressure/damper structural screen on four dev cases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import write_json,digest
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/flow_equivalent_short')
PAIRS=(('building_9001','denver_1'),('building_9002','atlanta_4'),('building_9000','phoenix_2'),('building_9002','singapore_2'))
ARMS=('phase_ctrl','flow_reset')


def prepare():
    files=list(Path('hvaclab/policies').glob('*.py'))+list(Path('smart_buildings').rglob('*.py'))
    files += [Path(p) for p in ('hvaclab/flow_equivalent_short.py','hvaclab/offhours_hold_worker.py','hvaclab/morning_recovery_worker.py','hvaclab/flow_equivalent_worker.py','hvaclab/flow_equivalent_review.py','hvaclab/morning_recovery_review.py','hvaclab/flow_equivalent_queue.py','hvaclab/plant_runtime_telemetry.py','hvaclab/phase_online_worker.py',
        'hvaclab/runner.py','hvaclab/registry.py','hvaclab/pipeline_entry.py','hvaclab/coverage.py','hvaclab/pi_calibration.py','hvaclab/supervisor.py')]
    for b,w in PAIRS:files.extend((Path('smart_buildings/holdout_scenarios/dev')/b/w).glob('*'))
    references={}
    for b,w in PAIRS:
        for key,old in (('phase_ctrl','phase_ctrl'),):
            path=Path('experiments/runs/plant_runtime_short/jobs')/f'{b}__{w}__{old}.result.json'
            if path.exists():references[f'{b}/{w}/{key}']=dict(path=path.as_posix(),sha256=digest(path))
    protocol=dict(steps=2016,pairs=PAIRS,arms=ARMS,max_workers=1,launch_interval_seconds=20,
        available_memory_gib_per_launch=1.25,base_policy='damper12_mid',normal_hold_steps=432,refresh_caps=[],pressure_floor_pa=200.,damper_floor=.1,target_opening=.98,trim_carried_memory=False,morning_window_seconds=7200,
        phases=['prestart','occupied','scheduled_empty','offhours'],
        objective='V10 controller with post-controller pressure/damper reparameterization preserving zone flows and all other actions; no online fitting or simulator truth inputs',
        definitions='prestart: PI window before comfort opens; occupied: comfort open and public occupants>0; scheduled_empty: comfort open and public occupants<=0; offhours: PI inactive',
        analysis='reward terms grouped by phase known at decision time; contributions normalized by total2016 steps; report post-step reward occupancy; thermal carryover means nonadditive effects',
        gate='mean loss at least1% below V10;5% material flag; mean comfort tolerance1e-6; morning max1e-6 or0.1% relative; physical flow errors at most2ppm; some actual intervention; no automatic full sweep/notebook',
        references=references,source_sha256={p.as_posix():digest(p) for p in sorted(set(files)) if p.is_file()},
        curve_reference_sha256=digest('experiments/runs/reheat_curve_calibration/rollouts.json'))
    protocol=json.loads(json.dumps(protocol));path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('inputs changed, preserve run directory')
    write_json(path,protocol);return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with RunLock(ROOT/'scheduler.lock'):
        protocol=prepare();fingerprint=digest(ROOT/'protocol.json');records=[];active=[];last_launch=-1e9
        from hvaclab.offhours_hold_worker import verify_record
        def checkpoint(state,waiting=False):
            write_json(ROOT/'status.json',dict(status=state,completed=len(records),total=len(PAIRS)*len(ARMS),active=[a[0]['task'] for a in active],waiting_for_memory=waiting,
                failed=[],updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            write_json(ROOT/'rollouts.json',dict(records=records))
        checkpoint('prepared')
        if args.prepare_only:return
        try:
            for stage in (ARMS,):
                pending=[]
                for arm in stage:
                    for b,w in PAIRS:
                        task=dict(building=b,weather=w,policy=arm,steps=2016,scenarios_dir='smart_buildings/holdout_scenarios/dev')
                        stem=f'{b}__{w}__{arm}';path=ROOT/'jobs'/(stem+'.job.json');result=ROOT/'jobs'/(stem+'.result.json')
                        job=dict(task=task,fingerprint=fingerprint,source_sha256=protocol['source_sha256'],result_path=str(result),reference=protocol['references'].get(f'{b}/{w}/{arm}'))
                        write_json(path,job)
                        if result.exists() and result.with_suffix('.thermal.json').exists() and 'flow_audit' in json.loads(result.read_text()):
                            record=json.loads(result.read_text());verify_record(record,job);records.append(record)
                        else:pending.append((job,path,result))
                while pending or active:
                    for job,process,log,result in list(active):
                        code=process.poll()
                        if code is None:continue
                        log.close();active.remove((job,process,log,result))
                        if not result.with_suffix('.thermal.json').exists() and not code:raise RuntimeError('missing thermal sidecar')
                        if code:raise RuntimeError('worker failed: '+str(result)+' exit '+str(code))
                        record=json.loads(result.read_text());verify_record(record,job);records.append(record)
                    waiting=bool(pending and len(active)<1 and psutil.virtual_memory().available<1.25*2**30)
                    if pending and len(active)<1 and not waiting and time.monotonic()-last_launch>=20:
                        job,path,result=pending.pop(0);log=path.with_suffix('.log').open('w')
                        process=subprocess.Popen([sys.executable,'-u','-m','hvaclab.flow_equivalent_worker','--job',str(path)],stdout=log,stderr=subprocess.STDOUT)
                        active.append((job,process,log,result));last_launch=time.monotonic()
                    checkpoint('running',waiting);time.sleep(5)
            checkpoint('complete');print('COMPLETE',len(records),flush=True)
        except BaseException as exc:
            for job,process,log,result in active:
                try:
                    proc=psutil.Process(process.pid)
                    for child in proc.children(recursive=True):child.terminate()
                    process.terminate();process.wait(timeout=10)
                except (psutil.NoSuchProcess,psutil.AccessDenied,subprocess.TimeoutExpired):pass
                log.close()
            active.clear();checkpoint('failed')
            write_json(ROOT/'failure.json',dict(error=str(exc)));raise


if __name__=='__main__':main()
