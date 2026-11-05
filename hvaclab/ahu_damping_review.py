"""Verify fixed four-arm damping results without promoting a short-screen winner."""
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.ahu_damping_verify import verify_damping
from hvaclab.flow_equivalent_validation import terms,morning_comparison
from hvaclab.morning_recovery_review import corrected_mornings

ROOT=Path('experiments/runs/ahu_damping_short')


def sensitivity(left,right):
    if left['schema']!=right['schema'] or len(left['rows'])!=len(right['rows']):raise ValueError('different sensitivity exposure')
    errors=[];mismatches=0;first=None
    for i,(a,b) in enumerate(zip(left['rows'],right['rows'])):
        if a['epoch']!=b['epoch']:raise ValueError('unmatched sensitivity clock')
        for zone,x in a['zones'].items():
            if any(x[f]!=b['zones'][zone][f] for f in ('occupancy','low','high')):raise ValueError('different occupancy/setpoint exposure')
        error=max(abs(x-y) for x,y in zip(a['observed_temperature'],b['observed_temperature']))
        errors.append(error);mismatches+=int(a['run']!=b['run'])
        if error>.1 and first is None:first=i
    return dict(max_temperature_difference=max(errors),mean_max_temperature_difference=float(np.mean(errors)),
                first_0_1K_step=first,ahu_run_mismatch_steps=mismatches,
                semantics='Difference induced by allocation with each feedback design; descriptive, not predictor error.')


def gate(means,cases):
    base=means['phase_ctrl'];flow=means['flow_reset'];combined=means['flow_smooth']
    active=all(row['diagnostics']['ahu_filtered_steps']>0 for case,row in cases['flow_smooth'].items() if not case.endswith('/denver_1'))
    morning=all(row['morning_vs_v10']['no_worse_each_morning'] for row in cases['flow_smooth'].values())
    worst=all(row['candidate']['loss']<=1.05*row['v10']['loss']+1e-10 for row in cases['flow_smooth'].values())
    comfort=combined['comfort']<=min(base['comfort'],flow['comfort'])+1e-6
    loss=combined['loss']<=.99*base['loss'] and combined['loss']<flow['loss']-1e-10
    return dict(active_on_both_regression_cases=active,morning_safe_vs_v10=morning,
                each_loss_within5pct_v10=worst,mean_comfort_no_worse_vs_both=comfort,
                mean_loss_pass=loss,passes_short_gate=bool(active and morning and worst and comfort and loss),
                change_vs_v10_pct=100*(combined['loss']/base['loss']-1),
                change_vs_flow_pct=100*(combined['loss']/flow['loss']-1))


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for name,sha in p['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('damping source changed '+name)
    records={};traces={}
    for path in (ROOT/'jobs').glob('*.result.json'):
        r=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        if r['fingerprint']!=fingerprint:raise ValueError('wrong damping protocol')
        verify_damping(r,job);t=r['task'];key=t['building'],t['weather'],t['policy']
        if key in records:raise ValueError('duplicate damping record')
        trace=json.loads(path.with_suffix('.thermal.json').read_text())
        if trace['fingerprint']!=fingerprint or trace['task']!=t or len(trace['rows'])!=4032:raise ValueError('incomplete thermal trace')
        rebuilt=corrected_mornings(trace,r['clock_semantics']['simulation_timestamp_naive'])
        if set(rebuilt)!=set(r['morning_comfort']):raise ValueError('wrong damping morning dates')
        for day,row in rebuilt.items():
            for field in ('first_occupied_epoch','steps','occupied_steps','prod_regret'):
                if abs(row[field]-r['morning_comfort'][day][field])>1e-8:raise ValueError('morning reconstruction mismatch')
        for field,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(x[field] for x in trace['rows'])-weight*r['result'][total])>1e-8:raise ValueError('thermal reward conservation failed')
        records[key]=r;traces[key]=trace
    if set(records)!={(b,w,a) for b,w in p['pairs'] for a in p['arms']}:raise ValueError('incomplete damping cohort')
    means={a:{k:sum(terms(records[b,w,a])[k] for b,w in p['pairs'])/len(p['pairs']) for k in ('loss','comfort','energy','carbon')} for a in p['arms']}
    cases={}
    for arm in p['arms'][1:]:
        cases[arm]={b+'/'+w:dict(v10=terms(records[b,w,'phase_ctrl']),candidate=terms(records[b,w,arm]),
            morning_vs_v10=morning_comparison(records[b,w,'phase_ctrl'],records[b,w,arm]),
            diagnostics=records[b,w,arm]['hold_diagnostics']) for b,w in p['pairs']}
    stability={b+'/'+w:dict(unfiltered=sensitivity(traces[b,w,'phase_ctrl'],traces[b,w,'flow_reset']),
                           filtered=sensitivity(traces[b,w,'ahu_smooth'],traces[b,w,'flow_smooth'])) for b,w in p['pairs']}
    result=dict(status='complete_verified',protocol_sha256=fingerprint,means=means,cases=cases,
                sensitivity=stability,gate=gate(means,cases),
                exact_v10_shadow_steps=sum(records[b,w,'phase_ctrl']['shadow_action_parity_steps'] for b,w in p['pairs']),
                full_sweep_started=False,notebook_exported=False,
                scope='Fixed14-day prefixes on two selected regressions and one heating control; reused cases, not independent or full-cohort validation.')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Bounded AHU damping factorial screen','',result['scope'],'',
           '| Arm | Loss | Comfort | Energy | Carbon |','|---|---:|---:|---:|---:|']
    for arm,row in means.items():lines.append('| '+arm+' | '+' | '.join(f'{row[k]:.8f}' for k in ('loss','comfort','energy','carbon'))+' |')
    lines+=['','No automatic promotion, notebook or full sweep follows from these prefixes. Filter bypasses zero/sign reversal/urgent raw error, and cannot alter AHU error by more than0.15K.','',
            '```json',json.dumps(result,indent=2),'```']
    Path('reports/AHU_DAMPING_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(means=means,gate=result['gate'],sensitivity=stability),indent=2))


if __name__=='__main__':main()
