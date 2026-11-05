"""Verify early-PI recovery against V10 with corrected simulator-clock mornings."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.offhours_hold_worker import verify_record
from hvaclab.morning_recovery_review import corrected_mornings
from hvaclab.coverage import valid_result

ROOT=Path('experiments/runs/morning_precool_short')


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
        trace=json.loads(path.with_suffix('.thermal.json').read_text())
        if trace['fingerprint']!=fingerprint or trace['task']!=t or len(trace['rows'])!=t['steps']:raise ValueError('invalid thermal trace')
        corrected=corrected_mornings(trace,r['clock_semantics']['simulation_timestamp_naive'])
        if set(corrected)!=set(r['morning_comfort']):raise ValueError('incorrect morning dates')
        for day,row in corrected.items():
            observed=r['morning_comfort'][day]
            for field in ('first_occupied_epoch','steps','occupied_steps','prod_regret'):
                if abs(row[field]-observed[field])>1e-8:raise ValueError('incorrect morning '+field)
        for source,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(x[source] for x in trace['rows'])-weight*r['result'][total])>1e-8:raise ValueError('trace reward conservation failed')
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
            if w=='atlanta_4':active=active and candidate['hold_diagnostics']['recovery_steps']>0
            arm_cases[b+'/'+w]=dict(v10=terms(control),candidate=terms(candidate),morning=morning,
                hold_audit=candidate['hold_audit'],baseline_hold_audit=control['hold_audit'],recovery_steps=candidate['hold_diagnostics']['recovery_steps'])
        score=means[arm]['loss'];baseline=means['phase_ctrl']['loss']
        gates[arm]=dict(early_recovery_active_in_atlanta=bool(active),morning_safe_in_both_cases=bool(safe),
            change_pct=100*(score/baseline-1),comfort_no_worse=means[arm]['comfort']<=means['phase_ctrl']['comfort']+1e-10,
            passes_short_gate=bool(active and safe and score<.99*baseline
                                  and means[arm]['comfort']<=means['phase_ctrl']['comfort']+1e-10))
        cases[arm]=arm_cases
    result=dict(status='complete_verified',means=means,cases=cases,gates=gates,
        exact_v10_shadow_steps=sum(records[b,w,'phase_ctrl']['shadow_action_parity_steps'] for b,w in p['pairs']),
        full_sweep_started=False,notebook_exported=False,
        scope='two selected previously used dev prefixes, daily morning guard; not hidden-score or unseen-building validation')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Warm-state early PI recovery - 2026-11-05','',
        'V10 controls and cap6 + temperature-gated one-hour PI extension. No truth feedback, no fitted coefficients, normal in-hours rules. Simulator-naive times retain their local wall clock; aware timestamps convert to building zone. Clock windows independently reconstructed from thermal traces.','',
        '| Arm | Mean prefix loss | Comfort | Energy | Carbon | Change vs V10 |', '|---|---:|---:|---:|---:|---:|']
    for arm,row in means.items():lines.append(f"| {arm} | {row['loss']:.8f} | {row['comfort']:.8f} | {row['energy']:.8f} | {row['carbon']:.8f} | {100*(row['loss']/means['phase_ctrl']['loss']-1):+.3f}% |")
    lines+=['','```json',json.dumps(result,indent=2),'```','',
        'Morning means refer to the first two reward-clock hours after the first observed occupied interval on each local date. Matched dates/exposure are mandatory; occasional empty intervals inside the window remain included. Worst occupied intervals are reported separately. These are selected partial scenarios, not full validation. No notebook exported.']
    Path('reports/MORNING_PRECOOL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(means=means,gates=gates,cases=cases),indent=2))


if __name__=='__main__':main()
