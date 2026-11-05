"""Attribute thermal-lift comfort regressions before freezing a guarded follow-up."""
import json
from pathlib import Path
import pandas as pd
from hvaclab.coverage import digest,write_json
from hvaclab.flow_comfort_review import weighted_discomfort

ROOT=Path('experiments/runs/cooling_lift_short')


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json');cases={}
    for name,sha in p['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('original lift sources changed; use frozen archive')
    for b,w in p['pairs']:
        paths=[ROOT/'jobs'/f'{b}__{w}__{arm}.result.thermal.json' for arm in ('phase_ctrl','flow_lift')]
        a,c=[json.loads(path.read_text()) for path in paths]
        if len(a['rows'])!=4032 or len(c['rows'])!=4032 or a['fingerprint']!=fingerprint or c['fingerprint']!=fingerprint:
            raise ValueError('incomplete or stale attribution traces')
        hot=cold=0.;hours={};zones={}
        for x,y in zip(a['rows'],c['rows']):
            if x['epoch']!=y['epoch'] or set(x['zones'])!=set(y['zones']):raise ValueError('unmatched attribution interval')
            if any(x['zones'][z]['occupancy']!=y['zones'][z]['occupancy'] for z in x['zones']):raise ValueError('occupancy exposure changed')
            pa,pb=weighted_discomfort(x['zones']),weighted_discomfort(y['zones'])
            hot+=(pb['hot']-pa['hot'])/4032;cold+=(pb['cold']-pa['cold'])/4032
            hour=pd.Timestamp(x['epoch'],unit='s').hour
            h=hours.setdefault(hour,dict(comfort_delta=0.,energy_carbon_delta=0.))
            h['comfort_delta']+=(y['comfort']-x['comfort'])/4032
            h['energy_carbon_delta']+=(y['energy']+y['carbon']-x['energy']-x['carbon'])/4032
        total=sum(y['comfort']-x['comfort'] for x,y in zip(a['rows'],c['rows']))/4032
        if abs(hot+cold-total)>2e-7:raise ValueError('hot/cold attribution does not conserve')
        cases[b+'/'+w]=dict(hot_delta=hot,cold_delta=cold,comfort_delta=total,
            hourly=hours,worst_hours=sorted(hours,key=lambda h:hours[h]['comfort_delta'],reverse=True)[:5],
            control_trace_sha256=digest(paths[0]),candidate_trace_sha256=digest(paths[1]))
    result=dict(status='complete_verified',original_protocol_sha256=fingerprint,analysis_sha256=digest(__file__),cases=cases,
        scope='Matched four4032-step prefixes; thermal direction/time attribution, not a causal estimate of a proposed guard.')
    write_json(ROOT/'comfort_attribution.json',result)
    lines=['# Cooling-lift comfort attribution','',result['scope'],'',
           '| Case | Hot comfort delta | Cold comfort delta | Total comfort delta |','|---|---:|---:|---:|']
    for name,r in cases.items():lines.append(f"| {name} | {r['hot_delta']:+.9f} | {r['cold_delta']:+.9f} | {r['comfort_delta']:+.9f} |")
    lines+=['','A uniform increase in cooling does not address hot and cold regressions simultaneously. The proposed follow-up uses only causal public-temperature history, with bounded signed planning-temperature projection and a separately measured protection gate.','',
            '```json',json.dumps(result,indent=2),'```']
    Path('reports/COOLING_LIFT_COMFORT_ATTRIBUTION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
