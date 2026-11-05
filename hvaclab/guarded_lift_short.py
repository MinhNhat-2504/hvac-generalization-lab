"""Signed planning-temperature projection and protected V10 fallback for cooling lift."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import write_json,digest
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/guarded_lift_short')
PAIRS=(('building_train_1','fairbanks_1'),('building_train_1','phoenix_5'),('building_9001','atlanta_4'),('building_9002','singapore_2'))
ARMS=('phase_ctrl','flow_lift','project_lift','guard_lift')


def prepare():
    files=list(Path('hvaclab/policies').glob('*.py'))+list(Path('smart_buildings').rglob('*.py'))
    files += [Path(p) for p in ('hvaclab/guarded_lift_short.py','hvaclab/offhours_hold_worker.py','hvaclab/morning_recovery_worker.py','hvaclab/flow_equivalent_worker.py','hvaclab/cooling_lift_worker.py','hvaclab/guarded_lift_worker.py','hvaclab/guarded_lift_verify.py','hvaclab/guarded_lift_review.py','hvaclab/cooling_lift_verify.py','hvaclab/cooling_lift_review.py','hvaclab/flow_equivalent_review.py','hvaclab/morning_recovery_review.py','hvaclab/guarded_lift_queue.py','hvaclab/plant_runtime_telemetry.py','hvaclab/phase_online_worker.py',
        'hvaclab/runner.py','hvaclab/registry.py','hvaclab/pipeline_entry.py','hvaclab/coverage.py','hvaclab/pi_calibration.py','hvaclab/supervisor.py')]
    for b,w in PAIRS:
        directory='smart_buildings/training_scenarios' if b.startswith('building_train_') else 'smart_buildings/holdout_scenarios/dev'
        files.extend((Path(directory)/b/w).glob('*'))
    references={}
    for b,w in PAIRS:
        for arm in ('phase_ctrl','flow_lift'):
            path=Path('experiments/runs/cooling_lift_short/jobs')/f'{b}__{w}__{arm}.result.thermal.json'
            references[f'{b}/{w}/{arm}']=dict(path=path.as_posix(),sha256=digest(path))
    protocol=dict(steps=4032,pairs=PAIRS,arms=ARMS,max_workers=2,launch_interval_seconds=20,
        available_memory_gib_per_launch=1.75,base_policy='damper12_mid',normal_hold_steps=432,refresh_caps=[],supply_reduction_k=2.,projection_steps=2.,projection_limit_k=.5,guard_margin_k=.25,pressure_floor_pa=200.,damper_floor=.1,target_opening=.98,trim_carried_memory=False,morning_window_seconds=7200,
        phases=['prestart','occupied','scheduled_empty','offhours'],
        objective='Bounded signed two-step publicT projection for allocation only; independent protected-window V10-handler fallback near forecast band crossings; preserve zonePI and36hhold',
        definitions='prestart: PI window before comfort opens; occupied: comfort open and public occupants>0; scheduled_empty: comfort open and public occupants<=0; offhours: PI inactive',
        analysis='reward terms grouped by phase known at decision time; contributions normalized by total4032 steps; report post-step reward occupancy; thermal carryover means nonadditive effects',
        gate='Each new arm needs>=5% lower meanloss vsV10; meancomfort<=V10+1e-6 and below original flow_lift; eachscenario loss<=V10*1.05; everymorning<=V10+max1e-6 or0.1%; lift active Phoenix/Singapore and signedprojection active; physical/model<=2ppm and noactualsupply guard violations; no automaticfullsweep/notebook',
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
        from hvaclab.guarded_lift_verify import verify_guarded as verify_record
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
                        task=dict(building=b,weather=w,policy=arm,steps=4032,scenarios_dir='smart_buildings/training_scenarios' if b.startswith('building_train_') else 'smart_buildings/holdout_scenarios/dev')
                        stem=f'{b}__{w}__{arm}';path=ROOT/'jobs'/(stem+'.job.json');result=ROOT/'jobs'/(stem+'.result.json')
                        job=dict(task=task,fingerprint=fingerprint,source_sha256=protocol['source_sha256'],result_path=str(result),prior_trace=protocol['references'].get(f'{b}/{w}/{arm}'))
                        write_json(path,job)
                        if result.exists() and result.with_suffix('.thermal.json').exists() and 'lift_audit' in json.loads(result.read_text()):
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
                    waiting=bool(pending and len(active)<2 and psutil.virtual_memory().available<1.75*2**30)
                    if pending and len(active)<2 and not waiting and time.monotonic()-last_launch>=20:
                        job,path,result=pending.pop(0);log=path.with_suffix('.log').open('w')
                        process=subprocess.Popen([sys.executable,'-u','-m','hvaclab.guarded_lift_worker','--job',str(path)],stdout=log,stderr=subprocess.STDOUT)
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
