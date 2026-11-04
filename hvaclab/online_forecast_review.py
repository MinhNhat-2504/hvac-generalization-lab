"""Verify completed online-transfer jobs and reproduce their local report."""
import json
from pathlib import Path
from hvaclab.coverage import digest,valid_result,write_json


def main():
    root=Path('experiments/runs/online_forecast_short');protocol=json.loads((root/'protocol.json').read_text())
    fingerprint=digest(root/'protocol.json')
    for path,sha in protocol['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('source changed: '+path)
    jobs={}
    for path in (root/'jobs').glob('*.result.json'):
        r=json.loads(path.read_text());task=r['task']
        if r['fingerprint']!=fingerprint or not valid_result(r['result'],task):raise ValueError('invalid rollout '+str(path))
        key=(task['building'],task['weather'],task['policy'])
        if key in jobs:raise ValueError('duplicate rollout')
        jobs[key]=r
    expected={(b,w,p) for b,w in protocol['pairs'] for p in protocol['policies']}
    if set(jobs)!=expected:raise ValueError('rollout cohort mismatch')
    cases=[]
    def terms(row):
        r=row['result'];n=r['steps']
        return dict(score=abs(r['mean_reward']),comfort=-.6*r['prod_regret']/n,
                    energy=.2*r['energy_cost']/n,carbon=.2*r['carbon']/n)
    parity=0
    for building,weather in protocol['pairs']:
        a=jobs[building,weather,'online_shadow'];b=jobs[building,weather,'online_quarter']
        if a['shadow_action_parity_steps']!=protocol['steps']:raise ValueError('shadow parity missing')
        parity+=a['shadow_action_parity_steps'];baseline=terms(a);candidate=terms(b)
        for row in (baseline,candidate):
            if abs(row['score']-sum(row[k] for k in ('comfort','energy','carbon')))>1e-8:raise ValueError('reward decomposition mismatch')
        cases.append(dict(building=building,weather=weather,v10=baseline,candidate=candidate,
            change_pct=100*(candidate['score']/baseline['score']-1),
            shadow_online=a['online_diagnostics'],feedback_online=b['online_diagnostics']))
    baseline={k:sum(r['v10'][k] for r in cases)/len(cases) for k in ('score','comfort','energy','carbon')}
    candidate={k:sum(r['candidate'][k] for r in cases)/len(cases) for k in baseline}
    gate=bool(all(r['feedback_online']['correction_zone_steps']>0 for r in cases)
              and candidate['score']<.99*baseline['score'] and candidate['comfort']<=baseline['comfort'])
    original=json.loads((root/'status.json').read_text())
    if original['status']!='complete' or original['decision']['passes_short_gate']!=gate:raise ValueError('checkpoint decision mismatch')
    result=dict(status='complete_verified',exact_v10_action_parity_steps=parity,cases=cases,
        mean_v10=baseline,mean_candidate=candidate,change_pct=100*(candidate['score']/baseline['score']-1),
        passes_short_gate=gate,full_sweep_started=False,notebook_exported=False,
        source_scope='two previously used dev buildings; partial 2016-step episodes; causal prequential evaluation, no hidden-score inference')
    write_json(root/'verified_review.json',result)
    lines=['# Causal online transfer and bounded PI feedback ? 2026-11-05','',
        'All four2016-step jobs verified against the frozen source/inputs and task fingerprints. Every shadow action matched V10:4032 steps. No static learned coefficients loaded; each job starts fresh. Error scores use predictions before the delayed target update and report only zone/state samples past the training minimum.','',
        '| Building/weather | V10 prefix loss | 25% feedback loss | Change |','|---|---:|---:|---:|']
    for r in cases:lines.append(f"| {r['building']}/{r['weather']} | {r['v10']['score']:.8f} | {r['candidate']['score']:.8f} | {r['change_pct']:+.3f}% |")
    lines+=['','```json',json.dumps(result,indent=2),'```','',
        'Online learning transfers to these two other buildings and improves observed forecast error, but the closed-loop reward gate fails: mean loss rises despite lower mean comfort loss. No large sweep or notebook is justified.',
        'The buildings have been used for previous dev studies; this is not independent validation. The result concerns two short prefixes, not full scenarios or Kaggle score.',
        'Next bounded diagnosis: separate PI compensation during prestart recovery from compensation while occupied; measure weighted comfort/energy/carbon for each before changing gains. Keep V10 fixed. Do not simply increase forecast blend on this evidence.']
    Path('reports/ONLINE_FORECAST_SHORT_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
