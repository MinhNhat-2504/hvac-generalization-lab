"""Complete-cohort, paired building-cluster analysis; never select on partial data."""
import json
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.paper_validation import ROOT,ARMS,WEATHER,check
from hvaclab.reserve_lift_verify import verify_reserve
from hvaclab.flow_equivalent_validation import terms,morning_comparison


def cluster_interval(base,candidate,draws=10000):
    a=np.asarray(base,dtype=float);b=np.asarray(candidate,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.all(np.isfinite(a+b)) or np.any(a<=0):
        raise ValueError('invalid building means')
    indices=np.random.default_rng(731).integers(0,len(a),size=(draws,len(a)))
    change=100*(b[indices].mean(axis=1)/a[indices].mean(axis=1)-1)
    return dict(mean_change_pct=float(100*(b.mean()/a.mean()-1)),
        cluster_95pct_interval=[float(v) for v in np.quantile(change,[.025,.975])],clusters=len(a),draws=draws)


def main():
    p=json.loads((ROOT/'protocol.json').read_text());check(p)
    cohort=json.loads((ROOT/'cohort.json').read_text());records={}
    for path,sha in cohort['input_sha256'].items():
        if digest(path)!=sha:raise ValueError('cohort input changed')
    for task in cohort['tasks']:
        stem='__'.join(task[k] for k in ('building','weather','policy'))
        job=json.loads((ROOT/'jobs'/(stem+'.job.json')).read_text())
        if job['fingerprint']!=digest(ROOT/'protocol.json') or job['task']!=task:raise ValueError('job provenance mismatch')
        r=json.loads((ROOT/'jobs'/(stem+'.result.json')).read_text());verify_reserve(r,job)
        records[task['building'],task['weather'],task['policy']]=r
    if len(records)!=192:raise ValueError('incomplete cohort')
    means={a:{k:float(np.mean([terms(records[b,w,a])[k] for b in cohort['buildings'] for w in WEATHER])) for k in ('loss','comfort','energy','carbon')} for a in ARMS}
    comparisons={}
    for arm in ARMS[1:]:
        base=[np.mean([terms(records[b,w,'phase_ctrl'])['loss'] for w in WEATHER]) for b in cohort['buildings']]
        candidate=[np.mean([terms(records[b,w,arm])['loss'] for w in WEATHER]) for b in cohort['buildings']]
        pairs=[]
        for b in cohort['buildings']:
            for w in WEATHER:
                a,c=records[b,w,'phase_ctrl'],records[b,w,arm]
                pairs.append(dict(building=b,weather=w,v10=terms(a),candidate=terms(c),morning=morning_comparison(a,c)))
        comparisons[arm]=dict(total_loss=cluster_interval(base,candidate),cases=pairs,
            failed_mornings=sum(len(v['morning']['failed_mornings']) for v in pairs),
            total_mornings=sum(v['morning']['days'] for v in pairs))
    write_json(ROOT/'verified_review.json',dict(status='complete_verified',protocol_sha256=digest(ROOT/'protocol.json'),
        means=means,comparisons=comparisons,scope=p['scope'],notebook_exported=False))
    print(json.dumps(dict(means=means,confidence={a:v['total_loss'] for a,v in comparisons.items()}),indent=2))


if __name__=='__main__':main()
