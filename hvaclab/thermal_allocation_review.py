"""Verify read-only allocation feasibility against independently saved V10 prefixes."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.offhours_hold_worker import verify_record

ROOT=Path('experiments/runs/thermal_allocation_audit_v2')


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for path,sha in p['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('audit source changed '+path)
    cases={}
    for b,w in p['pairs']:
        job=json.loads((ROOT/(w+'.job.json')).read_text());record=json.loads((ROOT/(w+'.result.json')).read_text())
        verify_record(record,job)
        allocation_path=ROOT/(w+'.result.allocation.json');allocation=json.loads(allocation_path.read_text())
        if allocation['fingerprint']!=fingerprint or len(allocation['rows'])!=864:raise ValueError('incomplete allocation observer')
        summary=allocation['summary']
        if not summary['controller_unchanged'] or not summary['reward_info_unchanged'] or not summary['per_device_system_meters_reconciled']:
            raise ValueError('read-only/meter audit failed')
        reference=Path('experiments/runs/comfort_band_short/jobs')/f'{b}__{w}__phase_ctrl.result.thermal.json'
        old=json.loads(reference.read_text())['rows'][:864]
        current=json.loads((ROOT/(w+'.result.thermal.json')).read_text())['rows']
        if len(old)!=864 or len(current)!=864:raise ValueError('missing V10 prefix')
        for a,c in zip(old,current):
            for field in ('epoch','observed_temperature','run','sat','damper','reheat'):
                if a[field]!=c[field]:raise ValueError('observer changed control prefix '+field)
        for source,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(r[source] for r in old)-weight*record['result'][total])>1e-8:raise ValueError('observer changed scorer term')
        if abs(summary['fan_Wh']-record['flow_audit']['actual_fan_Wh'])>max(.001,2e-6*summary['fan_Wh']):
            raise ValueError('independent physical fan-meter mismatch')
        candidates=[v for r in allocation['rows'] for v in r['cooling'].values()]
        electricity=summary['fan_Wh']+summary['cooling_Wh']
        beneficial=sum(max(0.,v['fan_saving_Wh']+v['cooling_saving_Wh']) for v in candidates)
        summary=dict(summary,beneficial_static_electricity_saving_Wh=beneficial,
                     beneficial_static_fraction_of_fan_ac=beneficial/max(1e-12,electricity),
                     reference_trace_sha256=digest(reference),allocation_sha256=digest(allocation_path))
        cases[b+'/'+w]=summary
    result=dict(status='complete_verified',protocol_sha256=fingerprint,analysis_sha256=digest(__file__),cases=cases,
                exact_v10_shadow_steps=sum(r['exact_v10_shadow_steps'] for r in cases.values()),
                initial_probe_excluded=True,controller_exported=False,
                scope='Three previously used3-day prefixes; static ideal feasibility only. No closed-loop candidate score or unseen-building validation.')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Thermal lift/airflow allocation feasibility','',result['scope'],'',
           'The initial meter-mapping probe is excluded. The corrected observer reads the aggregate scorer meter without inserting protobuf entries and reconciles per-device callbacks with system totals. Public actions and stored V10 prefixes match exactly.','',
           '| Case | Cooling candidates | Fan+AC kWh | Static beneficial kWh | Fraction | Ideal heating flow reduction |',
           '|---|---:|---:|---:|---:|---:|']
    for case,s in cases.items():lines.append(f"| {case} | {s['cooling_candidate_handler_steps']} | {(s['fan_Wh']+s['cooling_Wh'])/1000:.4f} | {s['beneficial_static_electricity_saving_Wh']/1000:.4f} | {100*s['beneficial_static_fraction_of_fan_ac']:.3f}% | {100*s['heat_weighted_ideal_flow_reduction']:.3f}% |")
    lines+=['','Heating figures assume unlimited coil-valve delivery and fixed pointwise net heat; they do not establish a feasible control action. Cooling figures require no reheat, feasible damper/fan floors and a2K supply reduction clipped at285K; they preserve current-state static heat, not implicit transient dynamics. They exclude pump and fuel effects and must not be described as total-score savings.','',
            '```json',json.dumps(result,indent=2),'```']
    Path('reports/THERMAL_ALLOCATION_FEASIBILITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
