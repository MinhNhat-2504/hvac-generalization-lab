"""Compare fixed band-PI arms against fresh centre-PI/flow baselines."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.comfort_band_verify import verify_band
from hvaclab.flow_equivalent_validation import terms,morning_comparison
from hvaclab.morning_recovery_review import corrected_mornings

ROOT=Path('experiments/runs/comfort_band_short')


def candidate_gate(arm,means,cases):
    base=means['phase_ctrl'];row=means[arm]
    active=all(r['diagnostics']['band_zone_steps']>0 for r in cases[arm].values())
    morning=all(r['morning_vs_v10']['no_worse_each_morning'] for r in cases[arm].values())
    worst=all(r['candidate']['loss']<=1.05*r['v10']['loss']+1e-10 for r in cases[arm].values())
    comfort=row['comfort']<=base['comfort']+1e-6
    loss=row['loss']<=.95*base['loss']
    flow_benchmark=arm!='band_flow' or row['loss']<=.99*means['flow_reset']['loss']
    return dict(change_vs_v10_pct=100*(row['loss']/base['loss']-1),
        active_all_cases=active,morning_safe=morning,each_loss_within5pct=worst,
        mean_comfort_no_worse=comfort,material5pct_loss_gain=loss,flow_benchmark=flow_benchmark,
        passes_short_gate=bool(active and morning and worst and comfort and loss and flow_benchmark))


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for name,sha in p['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('band PI source changed '+name)
    records={}
    for path in (ROOT/'jobs').glob('*.result.json'):
        r=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        if r['fingerprint']!=fingerprint:raise ValueError('wrong band PI protocol')
        verify_band(r,job);t=r['task'];key=t['building'],t['weather'],t['policy']
        if key in records:raise ValueError('duplicate band PI record')
        trace=json.loads(path.with_suffix('.thermal.json').read_text())
        if trace['fingerprint']!=fingerprint or trace['task']!=t or len(trace['rows'])!=4032:raise ValueError('incomplete thermal trace')
        rebuilt=corrected_mornings(trace,r['clock_semantics']['simulation_timestamp_naive'])
        if set(rebuilt)!=set(r['morning_comfort']):raise ValueError('wrong morning dates')
        for day,row in rebuilt.items():
            for field in ('first_occupied_epoch','steps','occupied_steps','prod_regret'):
                if abs(row[field]-r['morning_comfort'][day][field])>1e-8:raise ValueError('wrong morning exposure/regret')
        for field,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(x[field] for x in trace['rows'])-weight*r['result'][total])>1e-8:raise ValueError('reward conservation failed')
        records[key]=r
    if set(records)!={(b,w,a) for b,w in p['pairs'] for a in p['arms']}:raise ValueError('incomplete band PI cohort')
    means={a:{k:sum(terms(records[b,w,a])[k] for b,w in p['pairs'])/len(p['pairs']) for k in ('loss','comfort','energy','carbon')} for a in p['arms']}
    cases={a:{b+'/'+w:dict(v10=terms(records[b,w,'phase_ctrl']),candidate=terms(records[b,w,a]),
        morning_vs_v10=morning_comparison(records[b,w,'phase_ctrl'],records[b,w,a]),
        diagnostics=records[b,w,a]['hold_diagnostics']) for b,w in p['pairs']} for a in p['arms'][1:]}
    result=dict(status='complete_verified',protocol_sha256=fingerprint,means=means,cases=cases,
        gates={a:candidate_gate(a,means,cases) for a in ('band_pi','band_flow')},
        exact_v10_shadow_steps=sum(records[b,w,'phase_ctrl']['shadow_action_parity_steps'] for b,w in p['pairs']),
        full_sweep_started=False,notebook_exported=False,
        scope='Four fixed14-day prefixes, two training/two dev, all previously used in other studies; not independent full-cohort or hidden-score validation.')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Comfort-band PI factorial screen','',result['scope'],'',
        '| Arm | Loss | Comfort | Energy | Carbon |','|---|---:|---:|---:|---:|']
    for arm,row in means.items():lines.append('| '+arm+' | '+' | '.join(f'{row[k]:.8f}' for k in ('loss','comfort','energy','carbon'))+' |')
    lines+=['','No automatic notebook or full sweep. The5% mean-gain screen is fixed prospectively; retaining integral within the inner band is not an OFF command.','',
            '```json',json.dumps(result,indent=2),'```']
    Path('reports/COMFORT_BAND_PI_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(means=means,gates=result['gates']),indent=2))


if __name__=='__main__':main()
