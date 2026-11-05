"""Requested full-cohort measurement, retaining the failed short comfort gate."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock
from hvaclab.offhours_hold_worker import verify_record
from hvaclab.flow_equivalent_validation import fan_audit_passes

ROOT=Path('experiments/runs/flow_equivalent_full')
CACHE=Path('experiments/runs/damper_refinement/rollouts.json')
CONTROL_PAIRS=(('building_9001','denver_1'),('building_9002','atlanta_4'))


def split_for(building):
    return 'training' if building.startswith('building_train_') else 'dev'


def verify_full(record,job):
    verify_record(record,job)
    audit=record['flow_audit']
    if audit['checked_zone_steps']<=0 or audit['max_relative_actual_flow_error']>2e-6:
        raise ValueError('invalid full physical flow audit')
    if not fan_audit_passes(audit):raise ValueError('invalid full fan-meter audit')
    path=Path(job['result_path']).with_suffix('.thermal.json')
    trace=json.loads(path.read_text())
    if trace['task']!=job['task'] or trace['fingerprint']!=job['fingerprint'] or len(trace['rows'])!=8640:
        raise ValueError('invalid full thermal sidecar')


def prepare():
    from hvaclab.runner import shipped_pairs
    rows=[r for r in json.loads(CACHE.read_text())['results'] if r['policy']=='damper12_mid']
    by_pair={(r['building'],r['weather']):r for r in rows}
    if len(rows)!=50 or len(by_pair)!=50:raise ValueError('need exactly50 unique V10 references')
    for split,directory,n in (('training','smart_buildings/training_scenarios',30),('dev','smart_buildings/holdout_scenarios/dev',20)):
        expected=set(shipped_pairs(Path(directory)))
        selected={key for key in by_pair if split_for(key[0])==split}
        if expected!=selected or len(expected)!=n:raise ValueError('reference cohort mismatch '+split)
    for r in rows:
        from hvaclab.coverage import valid_result
        if not valid_result(r,dict(building=r['building'],weather=r['weather'],policy='damper12_mid',steps=8640)):
            raise ValueError('invalid full baseline')
    short=Path('experiments/runs/flow_equivalent_short/verified_review.json')
    short_result=json.loads(short.read_text())
    if short_result['status']!='complete_verified':raise ValueError('short cohort not verified')
    # The user requested full measurement. Preserve the short rejection rather
    # than reclassifying it as a passing screen or silently relaxing comfort.
    for b,w in by_pair:
        write_json(ROOT/'references'/f'{b}__{w}.json',dict(result=by_pair[b,w]))
    files=list(Path('hvaclab/policies').glob('*.py'))+list(Path('smart_buildings').rglob('*.py'))
    files += [Path(p) for p in ('hvaclab/flow_equivalent_full.py','hvaclab/flow_equivalent_full_review.py',
        'hvaclab/flow_equivalent_full_queue.py','hvaclab/flow_equivalent_package.py',
        'hvaclab/flow_equivalent_worker.py','hvaclab/flow_equivalent_validation.py',
        'hvaclab/offhours_hold_worker.py','hvaclab/morning_recovery_worker.py','hvaclab/morning_recovery_review.py',
        'hvaclab/phase_online_worker.py','hvaclab/plant_runtime_telemetry.py','hvaclab/coverage.py',
        'hvaclab/runner.py','hvaclab/registry.py','hvaclab/pipeline_entry.py','hvaclab/pi_calibration.py',
        'hvaclab/supervisor.py','hvaclab/submission/experimental.py','hvaclab/submission/build.py')]
    for directory in ('smart_buildings/training_scenarios','smart_buildings/holdout_scenarios/dev'):
        files+=list(Path(directory).rglob('*.gin'))+list(Path(directory).rglob('*.npy'))
    files+=list(Path('smart_buildings/weather/csv').glob('*.csv'))
    protocol=dict(steps=8640,baseline_policy='damper12_mid',candidate='flow_reset',
        baseline_sha256=digest(CACHE),short_review_sha256=digest(short),
        short_gate=short_result['gates']['flow_reset'],
        reason='User requested full training/dev measurement after short total loss improvement but failed comfort guard; no automatic promotion from the short run',
        dependency_versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','gin-config','protobuf')},
        baseline_rows=rows,control_pairs=CONTROL_PAIRS,max_workers=2,memory_gib_per_launch=1.75,
        launch_interval_seconds=20,source_sha256={p.as_posix():digest(p) for p in sorted(set(files))},
        curve_reference_sha256=digest('experiments/runs/reheat_curve_calibration/rollouts.json'),
        packaging_gate='Both full means strictly below V10; mean comfort each split no worse within1e-6. Only if these pass, run remaining48 full V10 controls to check all50 paired daily mornings within max1e-6 or0.1%; physical2ppm flow audit; exact standalone action parity. Report all scenario regressions.')
    protocol=json.loads(json.dumps(protocol));path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('full inputs changed; preserve run')
    write_json(path,protocol);return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    with RunLock(ROOT/'scheduler.lock'):
        p=prepare();fingerprint=digest(ROOT/'protocol.json');completed=[];active=[];last_launch=-1e9;stage='prepared';target_jobs=52
        def checkpoint(status,waiting=False):
            write_json(ROOT/'status.json',dict(status=status,stage=stage,completed=len(completed),total=target_jobs,
                counts={s:sum(r['task']['policy']=='flow_reset' and r['task']['split']==s for r in completed) for s in ('training','dev')},
                active=[x[0]['task'] for x in active],waiting_for_memory=waiting,failed=[],updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        checkpoint('prepared')
        if args.prepare_only:return
        phases=[('reference_controls',[(split_for(b),b,w,'phase_ctrl') for b,w in CONTROL_PAIRS]),
                ('full_cohorts',[(split_for(r['building']),r['building'],r['weather'],'flow_reset') for r in sorted(p['baseline_rows'],key=lambda r:(split_for(r['building'])!='dev',r['weather'],r['building']))])]
        try:
            for stage,tasks in phases:
                pending=[]
                for split,b,w,arm in tasks:
                    task=dict(split=split,building=b,weather=w,policy=arm,steps=8640,
                              scenarios_dir='smart_buildings/training_scenarios' if split=='training' else 'smart_buildings/holdout_scenarios/dev')
                    stem=f'{split}__{b}__{w}__{arm}';descriptor=ROOT/'jobs'/(stem+'.job.json');result=ROOT/'jobs'/(stem+'.result.json')
                    ref=ROOT/'references'/f'{b}__{w}.json'
                    job=dict(task=task,result_path=str(result),fingerprint=fingerprint,source_sha256=p['source_sha256'],
                             reference=dict(path=ref.as_posix(),sha256=digest(ref)) if arm=='phase_ctrl' else None)
                    write_json(descriptor,job)
                    if result.exists() and result.with_suffix('.thermal.json').exists() and 'flow_audit' in json.loads(result.read_text()):
                        record=json.loads(result.read_text());verify_full(record,job);completed.append(record)
                    else:pending.append((job,descriptor,result))
                while pending or active:
                    for job,proc,log,result in list(active):
                        code=proc.poll()
                        if code is None:continue
                        log.close();active.remove((job,proc,log,result))
                        if code:raise RuntimeError('full worker failed '+str(result))
                        record=json.loads(result.read_text());verify_full(record,job);completed.append(record)
                    waiting=bool(pending and len(active)<2 and psutil.virtual_memory().available<1.75*2**30)
                    if pending and len(active)<2 and not waiting and time.monotonic()-last_launch>=20:
                        job,descriptor,result=pending.pop(0);log=descriptor.with_suffix('.log').open('w')
                        proc=subprocess.Popen([sys.executable,'-u','-m','hvaclab.flow_equivalent_worker','--job',str(descriptor)],stdout=log,stderr=subprocess.STDOUT)
                        active.append((job,proc,log,result));last_launch=time.monotonic()
                    checkpoint('running',waiting);time.sleep(5)
                if stage=='full_cohorts':
                    from hvaclab.flow_equivalent_full_review import full_mean_gate
                    candidate=[r['result'] for r in completed if r['task']['policy']=='flow_reset']
                    if full_mean_gate(candidate,p['baseline_rows']):
                        target_jobs=100
                        phases.append(('morning_confirmation',[(split_for(r['building']),r['building'],r['weather'],'phase_ctrl')
                            for r in p['baseline_rows'] if (r['building'],r['weather']) not in CONTROL_PAIRS]))
            stage='finished';checkpoint('complete')
        except BaseException:
            for job,proc,log,result in active:
                try:
                    parent=psutil.Process(proc.pid)
                    for child in reversed(parent.children(recursive=True)):child.terminate()
                    parent.terminate();proc.wait(timeout=10)
                except (psutil.NoSuchProcess,psutil.AccessDenied,subprocess.TimeoutExpired):pass
                log.close()
            active.clear();checkpoint('failed');raise


if __name__=='__main__':main()
