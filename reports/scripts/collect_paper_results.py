"""Read-only private collection of the completed Fitlab paper experiment.

Copy outside hvaclab/; it must not change the frozen source enumeration.
No credentials, home-directory scan, paper or virtual environment is collected.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path,PurePosixPath
import platform
import subprocess
import sys
import time
import zipfile

RUN='experiments/runs/paper_confirmation_20261106'
ARMS=('phase_ctrl','flow_only','project_lift','reserve20')
WEATHER=('fairbanks_1','phoenix_5','atlanta_4','singapore_2')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def contained(repo,name):
    path=(repo/name).resolve()
    if not path.is_relative_to(repo.resolve()):raise ValueError('file outside repository: '+str(name))
    if not path.is_file():raise ValueError('missing file: '+str(name))
    return path


def check_design(protocol,cohort):
    if protocol.get('steps')!=8640 or protocol.get('building_count')!=12:
        raise ValueError('incorrect episode/cohort design')
    if tuple(protocol.get('arms',[]))!=ARMS or tuple(protocol.get('weather',[]))!=WEATHER:
        raise ValueError('incorrect arms/weather')
    buildings=cohort['buildings']
    if len(buildings)!=12 or len(set(buildings))!=12:raise ValueError('duplicate or missing buildings')
    expected={(b,w,a) for b in buildings for w in WEATHER for a in ARMS}
    keys=[]
    for task in cohort['tasks']:
        if task['steps']!=8640:raise ValueError('short episode')
        keys.append(tuple(task[k] for k in ('building','weather','policy')))
    if len(keys)!=192 or set(keys)!=expected:raise ValueError('incomplete or duplicate task grid')


def check_completion(state,review,fingerprint):
    if state.get('status')!='complete' or state.get('completed')!=192 or state.get('total')!=192 or state.get('active'):
        raise ValueError('192 completed jobs required; collection is not a partial-results export')
    if review.get('status')!='complete_verified' or review.get('protocol_sha256')!=fingerprint:
        raise ValueError('verified server review missing or belongs to another protocol')


def environment(repo):
    packages={}
    for d in importlib.metadata.distributions():
        name=d.metadata.get('Name')
        if name:packages[name]=d.version
    # Capture names/versions, not pip URLs, environment variables or credentials.
    data=dict(python=sys.version,platform=dict(system=platform.system(),release=platform.release(),machine=platform.machine()),
        packages=dict(sorted(packages.items())),collected_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        import psutil
        data['cpu_count_visible']=len(psutil.Process().cpu_affinity())
        data['ram_available_gib']=psutil.virtual_memory().available/2**30
    except (ImportError,AttributeError,PermissionError):pass
    for name,path in (('cpu_quota','/sys/fs/cgroup/cpu.max'),('memory_quota','/sys/fs/cgroup/memory.max')):
        p=Path(path)
        if p.is_file():data[name]=p.read_text().strip()
    if (repo/'release_manifest.json').is_file():data['release_commit']=read(repo/'release_manifest.json').get('commit')
    if (repo/'.git').exists():
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
        if not result.returncode:data['current_git_commit']=result.stdout.strip()
    return data


def audit(repo,include_traces):
    root=repo/RUN;p=read(root/'protocol.json');cohort=read(root/'cohort.json');review=read(root/'verified_review.json')
    fingerprint=sha(root/'protocol.json');check_design(p,cohort);check_completion(read(root/'status.json'),review,fingerprint)
    files={RUN+'/'+n for n in ('protocol.json','cohort.json','verified_review.json','status.json')}
    inputs={**p['source_sha256'],**cohort['input_sha256']}
    for name,digest in inputs.items():
        if sha(contained(repo,name))!=digest:raise ValueError('source/scenario hash mismatch: '+name)
    # Imports are from the frozen release, with no modifications to its files.
    sys.path.insert(0,str(repo))
    from hvaclab.reserve_lift_verify import verify_reserve
    from hvaclab.flow_equivalent_validation import terms,morning_comparison
    from hvaclab.paper_validation_review import cluster_interval
    from hvaclab.morning_recovery_review import corrected_mornings
    records={};traces={}
    for i,task in enumerate(cohort['tasks']):
        stem='__'.join(task[k] for k in ('building','weather','policy'))
        jobname=RUN+'/jobs/'+stem+'.job.json';resultname=RUN+'/jobs/'+stem+'.result.json'
        trace_name=RUN+'/jobs/'+stem+'.result.thermal.json'
        job=read(contained(repo,jobname));r=read(contained(repo,resultname))
        if job['task']!=task or job['fingerprint']!=fingerprint or job['source_sha256']!=p['source_sha256']:
            raise ValueError('job identity/source mismatch: '+stem)
        verify_reserve(r,job)
        trace_path=contained(repo,trace_name);trace=read(trace_path)
        if trace['task']!=task or trace['fingerprint']!=fingerprint or len(trace['rows'])!=8640:
            raise ValueError('trace identity/count mismatch: '+stem)
        rows=trace['rows']
        if any(y['epoch']-x['epoch']!=300 for x,y in zip(rows,rows[1:])):raise ValueError('trace clock gap')
        for field,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if not math.isclose(sum(x[field] for x in rows),weight*r['result'][total],rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError('trace reward mismatch: '+stem+'/'+field)
        mornings=corrected_mornings(trace,r['clock_semantics']['simulation_timestamp_naive'])
        if set(mornings)!=set(r['morning_comfort']):raise ValueError('trace morning dates mismatch')
        for day,values in mornings.items():
            for key,value in values.items():
                other=r['morning_comfort'][day][key]
                if isinstance(value,str):
                    if value!=other:raise ValueError('trace local clock mismatch')
                elif not math.isclose(value,other,rel_tol=1e-10,abs_tol=1e-8):raise ValueError('trace morning mismatch')
        key=tuple(task[k] for k in ('building','weather','policy'));records[key]=r
        traces[trace_name]=dict(sha256=sha(trace_path),bytes=trace_path.stat().st_size,included=include_traces)
        files.update((jobname,resultname))
        if include_traces:files.add(trace_name)
        logname=RUN+'/jobs/'+stem+'.job.log'
        if (repo/logname).is_file():files.add(logname)
        if (i+1)%16==0:print('AUDITED',i+1,'/192',flush=True)
    import numpy as np
    for arm in ARMS:
        for metric in ('loss','comfort','energy','carbon'):
            value=np.mean([terms(records[b,w,arm])[metric] for b in cohort['buildings'] for w in WEATHER])
            if not math.isclose(value,review['means'][arm][metric],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('review mean mismatch')
        if arm=='phase_ctrl':continue
        actual=review['comparisons'][arm];pairs={}
        for case in actual['cases']:
            k=case['building'],case['weather']
            if k in pairs:raise ValueError('duplicate review case')
            pairs[k]=case
        if set(pairs)!={(b,w) for b in cohort['buildings'] for w in WEATHER}:raise ValueError('incomplete review cases')
        failures=days=0
        for b,w in pairs:
            a,c=records[b,w,'phase_ctrl'],records[b,w,arm];row=pairs[b,w]
            if row['v10']!=terms(a) or row['candidate']!=terms(c):raise ValueError('review paired values mismatch')
            m=morning_comparison(a,c)
            if m!=row['morning']:raise ValueError('review paired mornings mismatch')
            failures+=len(m['failed_mornings']);days+=m['days']
        if failures!=actual['failed_mornings'] or days!=actual['total_mornings']:raise ValueError('review exposure mismatch')
        baseline=[np.mean([terms(records[b,w,'phase_ctrl'])['loss'] for w in WEATHER]) for b in cohort['buildings']]
        proposed=[np.mean([terms(records[b,w,arm])['loss'] for w in WEATHER]) for b in cohort['buildings']]
        ci=cluster_interval(baseline,proposed)
        if ci!=actual['total_loss']:raise ValueError('review cluster interval mismatch')
    if (repo/RUN/'server.log').is_file():files.add(RUN+'/server.log')
    for name in ('release_manifest.json','pilot_data_manifest.json'):
        if (repo/name).is_file():files.add(name)
    for pilot in sorted(root.glob('preflight_*')):
        if pilot.is_dir() and (pilot/'verified.json').is_file():files.add((pilot/'verified.json').relative_to(repo).as_posix())
    return files,dict(protocol_sha256=fingerprint,status='complete_audited',jobs=192,
        traces=traces,frozen_inputs_sha256=inputs,scope=p['scope'],include_traces=include_traces)


def safe_member(name):
    path=PurePosixPath(name)
    return bool(name and '\\' not in name and not path.is_absolute() and '..' not in path.parts and ':' not in name)


def verify_bundle(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or not all(safe_member(n) for n in names):raise ValueError('unsafe or duplicate ZIP entries')
        manifest=json.loads(z.read('collection_manifest.json'))
        if set(names)!=set(manifest['files_sha256'])|{'collection_manifest.json'}:raise ValueError('unexpected or missing ZIP files')
        for name,digest in manifest['files_sha256'].items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=digest:raise ValueError('ZIP content hash mismatch: '+name)
    return manifest


def collect(repo,output,include_traces=False):
    repo=repo.resolve()
    if output.exists():raise ValueError('output exists; preserve it or choose a different output name')
    # Relative job paths in the existing verifier resolve against repository cwd.
    import os
    previous=Path.cwd();os.chdir(repo)
    try:files,audit_data=audit(repo,include_traces)
    finally:os.chdir(previous)
    env=environment(repo);output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp')
    if temp.exists():raise ValueError('temporary output exists; inspect previous collection first')
    manifest=dict(format_version=1,private_transfer_only=True,files_sha256={})
    try:
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                if not safe_member(name):raise ValueError('unsafe archive member')
                h=hashlib.sha256()
                with contained(repo,name).open('rb') as src,z.open(name,'w',force_zip64=True) as dst:
                    for chunk in iter(lambda:src.read(1024*1024),b''):dst.write(chunk);h.update(chunk)
                manifest['files_sha256'][name]=h.hexdigest()
            for name,data in (('collection_audit.json',audit_data),('environment.json',env)):
                blob=json.dumps(data,indent=2).encode();z.writestr(name,blob);manifest['files_sha256'][name]=hashlib.sha256(blob).hexdigest()
            z.writestr('collection_manifest.json',json.dumps(manifest,indent=2))
        verify_bundle(temp);temp.replace(output)
    except BaseException:
        # Keep an incomplete ZIP for inspection; never delete source results.
        raise
    print('PRIVATE_RESULTS_BUNDLE_VERIFIED',str(output),'SHA256',sha(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path)
    parser.add_argument('--include-traces',action='store_true')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--verify-bundle',type=Path)
    args=parser.parse_args()
    if args.verify_bundle:
        m=verify_bundle(args.verify_bundle);print('BUNDLE_HASHES_VERIFIED',len(m['files_sha256']),'files');return
    if args.output is None:parser.error('--output required for collection')
    repo=args.repo.resolve();output=args.output.resolve()
    while args.wait:
        try:
            state=read(repo/RUN/'status.json');review=read(repo/RUN/'verified_review.json')
            check_completion(state,review,sha(repo/RUN/'protocol.json'));break
        except (OSError,ValueError,KeyError):
            print('WAITING_FOR_192_AND_VERIFIED_REVIEW',flush=True);time.sleep(30)
    collect(repo,output,args.include_traces)


if __name__=='__main__':main()
