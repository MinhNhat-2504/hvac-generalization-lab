"""Verify hold-refresh caps against fixed V10 and daily morning comfort guards."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.offhours_hold_worker import verify_record

ROOT=Path('experiments/runs/offhours_hold_short')


def terms(record):
    r=record['result'];n=r['steps']
    return dict(loss=abs(r['mean_reward']),comfort=-.6*r['prod_regret']/n,energy=.2*r['energy_cost']/n,carbon=.2*r['carbon']/n)


def morning_comparison(control,candidate):
    base=control['morning_comfort'];other=candidate['morning_comfort'];failed=[];worst=[]
    if set(base)!=set(other):raise ValueError('different local morning dates')
    for day,a in base.items():
        b=other[day]
        for field in ('first_occupied_epoch','steps','occupied_steps'):
            if a[field]!=b[field]:raise ValueError('different morning exposure '+day+'/'+field)
        baseline=-.6*a['prod_regret']/a['steps'];proposed=-.6*b['prod_regret']/b['steps']
        if proposed>baseline+1e-8:failed.append(dict(day=day,baseline=baseline,candidate=proposed,delta=proposed-baseline))
        if b['worst_occupied_comfort']>a['worst_occupied_comfort']+1e-8:
            worst.append(dict(day=day,baseline=a['worst_occupied_comfort'],candidate=b['worst_occupied_comfort']))
    return dict(days=len(base),no_worse_each_morning=not failed,failed_mornings=failed,worst_interval_regressions=worst)


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for path,sha in p['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('source changed '+path)
    records={}
    for path in (ROOT/'jobs').glob('*.result.json'):
        r=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        if r['fingerprint']!=fingerprint:raise ValueError('wrong protocol')
        verify_record(r,job);t=r['task'];key=(t['building'],t['weather'],t['policy'])
        if key in records:raise ValueError('duplicate record')
        records[key]=r
    if set(records)!={(b,w,a) for b,w in p['pairs'] for a in p['arms']}:raise ValueError('incomplete cohort')
    means={};cases={};gates={}
    for arm in p['arms']:
        rows=[records[b,w,arm] for b,w in p['pairs']]
        means[arm]={k:sum(terms(r)[k] for r in rows)/len(rows) for k in ('loss','comfort','energy','carbon')}
    for arm in p['arms'][1:]:
        arm_cases={};active=True;safe=True
        for b,w in p['pairs']:
            control=records[b,w,'phase_ctrl'];candidate=records[b,w,arm]
            morning=morning_comparison(control,candidate);safe=safe and morning['no_worse_each_morning']
            active=active and candidate['hold_audit']['capped_refreshes']>0
            arm_cases[b+'/'+w]=dict(v10=terms(control),candidate=terms(candidate),morning=morning,
                hold_audit=candidate['hold_audit'],baseline_hold_audit=control['hold_audit'])
        score=means[arm]['loss'];baseline=means['phase_ctrl']['loss']
        gates[arm]=dict(active_in_both_cases=bool(active),morning_safe_in_both_cases=bool(safe),
            change_pct=100*(score/baseline-1),comfort_no_worse=means[arm]['comfort']<=means['phase_ctrl']['comfort']+1e-10,
            passes_short_gate=bool(active and safe and score<.99*baseline
                                  and means[arm]['comfort']<=means['phase_ctrl']['comfort']+1e-10))
        cases[arm]=arm_cases
    result=dict(status='complete_verified',means=means,cases=cases,gates=gates,
        exact_v10_shadow_steps=sum(records[b,w,'phase_ctrl']['shadow_action_parity_steps'] for b,w in p['pairs']),
        full_sweep_started=False,notebook_exported=False,
        scope='two selected previously used dev prefixes, daily morning guard; not hidden-score or unseen-building validation')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Empty offhours hold-refresh caps - 2026-11-05','',
        'V10 foundation; cap only new hold refreshes when public occupancy is zero and PI window inactive. No trim of carried hold memory, no forced shutdown of inherited requests, no online feedback. Protected rules retain432-step refresh; state may differ after intervention.','',
        '| Arm | Mean prefix loss | Comfort | Energy | Carbon | Change vs V10 |', '|---|---:|---:|---:|---:|---:|']
    for arm,row in means.items():lines.append(f"| {arm} | {row['loss']:.8f} | {row['comfort']:.8f} | {row['energy']:.8f} | {row['carbon']:.8f} | {100*(row['loss']/means['phase_ctrl']['loss']-1):+.3f}% |")
    lines+=['','```json',json.dumps(result,indent=2),'```','',
        'Morning means refer to the first two reward-clock hours after the first observed occupied interval on each local date. Matched dates/exposure are mandatory; occasional empty intervals inside the window remain included. Worst occupied intervals are reported separately. These are selected partial scenarios, not full validation. No notebook exported.']
    Path('reports/OFFHOURS_HOLD_REFRESH_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(means=means,gates=gates,cases=cases),indent=2))


if __name__=='__main__':main()
