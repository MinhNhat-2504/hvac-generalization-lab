"""Compare all complete training/dev episodes and gate standalone packaging."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.flow_equivalent_validation import terms,morning_comparison

ROOT=Path('experiments/runs/flow_equivalent_full')


def cohort_statistics(candidate,baseline):
    by_pair={(r['building'],r['weather']):r for r in candidate}
    refs={(r['building'],r['weather']):r for r in baseline}
    if len(candidate)!=len(by_pair) or set(by_pair)!=set(refs):raise ValueError('incomplete/duplicate full cohort')
    if any(r['steps']!=8640 or r.get('error') for r in candidate):raise ValueError('short or failed full episode')
    n=len(refs)
    a={k:sum(terms(dict(result=r))[k] for r in baseline)/n for k in ('loss','comfort','energy','carbon')}
    b={k:sum(terms(dict(result=r))[k] for r in candidate)/n for k in a}
    comparisons=[dict(building=x,weather=w,v10=abs(refs[x,w]['mean_reward']),candidate=abs(r['mean_reward']),
                      delta=abs(r['mean_reward'])-abs(refs[x,w]['mean_reward'])) for (x,w),r in by_pair.items()]
    return dict(n=n,v10=a,candidate=b,change_pct=100*(b['loss']/a['loss']-1),
                passes_mean_gate=bool(b['loss']<a['loss']-1e-10 and b['comfort']<=a['comfort']+1e-6),
                improved=sum(r['delta']<-1e-10 for r in comparisons),regressed=sum(r['delta']>1e-10 for r in comparisons),
                comparisons=sorted(comparisons,key=lambda r:r['delta'],reverse=True))


def full_mean_gate(candidate,baseline):
    for training,n in ((True,30),(False,20)):
        a=[r for r in candidate if r['building'].startswith('building_train_')==training]
        b=[r for r in baseline if r['building'].startswith('building_train_')==training]
        if len(a)!=n or len(b)!=n:raise ValueError('full mean gate requires30 training and20 dev episodes')
        if not cohort_statistics(a,b)['passes_mean_gate']:return False
    return True


def main():
    from hvaclab.flow_equivalent_full import verify_full,split_for,CACHE
    from hvaclab.morning_recovery_review import corrected_mornings
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    if digest(CACHE)!=p['baseline_sha256']:raise ValueError('baseline cache changed')
    for name,sha in p['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('full source changed '+name)
    records={}
    for path in (ROOT/'jobs').glob('*.result.json'):
        r=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        if r['fingerprint']!=fingerprint:raise ValueError('wrong full protocol')
        verify_full(r,job);t=r['task'];key=t['building'],t['weather'],t['policy']
        if key in records:raise ValueError('duplicate full record')
        trace=json.loads(path.with_suffix('.thermal.json').read_text())
        for source,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(x[source] for x in trace['rows'])-weight*r['result'][total])>1e-8:raise ValueError('full trace reward mismatch')
        rebuilt=corrected_mornings(trace,r['clock_semantics']['simulation_timestamp_naive'])
        if set(rebuilt)!=set(r['morning_comfort']):raise ValueError('full morning date mismatch')
        for day,row in rebuilt.items():
            for field in ('first_occupied_epoch','steps','occupied_steps','prod_regret'):
                if abs(row[field]-r['morning_comfort'][day][field])>1e-8:raise ValueError('full morning reconstruction mismatch')
        records[key]=r
    baseline=p['baseline_rows'];expected={(r['building'],r['weather']) for r in baseline}
    candidate=[r['result'] for (b,w,a),r in records.items() if a=='flow_reset']
    if {(r['building'],r['weather']) for r in candidate}!=expected:raise ValueError('missing full candidates')
    statistics={s:cohort_statistics([r for r in candidate if split_for(r['building'])==s],
                                   [r for r in baseline if split_for(r['building'])==s]) for s in ('training','dev')}
    means_pass=all(s['passes_mean_gate'] for s in statistics.values())
    controls={(b,w):r for (b,w,a),r in records.items() if a=='phase_ctrl'}
    morning={b+'/'+w:morning_comparison(control,records[b,w,'flow_reset']) for (b,w),control in controls.items()}
    all_mornings=len(controls)==50 and all(m['no_worse_each_morning'] for m in morning.values())
    result=dict(status='complete_verified',protocol_sha256=fingerprint,statistics=statistics,
        short_gate=p['short_gate'],full_mean_gate=means_pass,paired_full_morning_cases=len(controls),
        all50_morning_gate=all_mornings,morning=morning,eligible_for_packaging=bool(means_pass and all_mornings),
        baseline_control_parity_steps=sum(r['shadow_action_parity_steps'] for r in controls.values()),
        notebook_exported=False,scope='complete local30training+20dev episodes; no organizer-score claim')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Full flow-equivalent validation','',result['scope'],'',
           '| Split | Cases | V10 | Candidate | Change | Comfort mean gate |', '|---|---:|---:|---:|---:|---|']
    for s,row in statistics.items():lines.append(f"| {s} | {row['n']} | {row['v10']['loss']:.8f} | {row['candidate']['loss']:.8f} | {row['change_pct']:+.3f}% | {row['passes_mean_gate']} |")
    lines+=['','The failed short comfort gate is retained. Remaining48 baseline morning traces are run only if both full mean gates pass. No incomplete morning cohort is treated as all50 validated.','',
            '```json',json.dumps(result,indent=2),'```']
    Path('reports/FLOW_EQUIVALENT_FULL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('morning','statistics')},indent=2))


if __name__=='__main__':main()
