"""Compare actual reward-clock morning temperatures without feeding truth back."""
import json
from pathlib import Path
import ast
import re
import numpy as np
import pandas as pd
from hvaclab.coverage import digest, write_json
from hvaclab.offhours_hold_review import morning_comparison, terms

ROOT = Path('experiments/runs/morning_diagnosis_short')


def corrected_mornings(trace, naive_clock):
    """Respect environment semantics: naive simulator time is already local."""
    days={}
    for row in trace['rows']:
        clock=pd.Timestamp(row['epoch'],unit='s') if naive_clock else pd.Timestamp(row['epoch'],unit='s',tz='UTC').tz_convert(trace['schema']['time_zone'])
        day=str(clock.date());occupied=sum(z['occupancy'] for z in row['zones'].values())>0
        if day not in days and occupied:
            days[day]=dict(first_occupied_epoch=row['epoch'],first_local_clock=str(clock),steps=0,occupied_steps=0,
                           prod_regret=0.,worst_occupied_comfort=0.)
        if day not in days or not 0<=row['epoch']-days[day]['first_occupied_epoch']<7200:
            continue
        target=days[day];target['steps']+=1;target['occupied_steps']+=int(occupied)
        target['prod_regret']+=row['comfort']/(-.6)
        if occupied:target['worst_occupied_comfort']=max(target['worst_occupied_comfort'],row['comfort'])
    return days


def summarize(rows):
    if not rows:
        return dict(steps=0)
    occupants = sum(z['occupancy'] for r in rows for z in r['zones'].values())
    cold = sum(z['occupancy'] * max(0., z['low'] - z['temperature'])
               for r in rows for z in r['zones'].values())
    hot = sum(z['occupancy'] * max(0., z['temperature'] - z['high'])
              for r in rows for z in r['zones'].values())
    return dict(steps=len(rows), comfort_mean=sum(r['comfort'] for r in rows)/len(rows),
                energy_carbon_mean=sum(r['energy']+r['carbon'] for r in rows)/len(rows),
                mean_temperature=float(np.mean([z['temperature'] for r in rows for z in r['zones'].values()])),
                mean_cold_deviation=float(np.mean([max(0., z['low']-z['temperature']) for r in rows for z in r['zones'].values()])),
                mean_hot_deviation=float(np.mean([max(0., z['temperature']-z['high']) for r in rows for z in r['zones'].values()])),
                occupied_cold_degree_person_steps=cold, occupied_hot_degree_person_steps=hot,
                occupant_steps=occupants, ahu_machine_hours=sum(sum(r['run']) for r in rows)/12,
                mean_damper=float(np.mean([(x+1)/2 for r in rows for x in r['damper']])),
                mean_reheat=float(np.mean([(x+1)/2 for r in rows for x in r['reheat']])))


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for path, sha in protocol['source_sha256'].items():
        if digest(path)!=sha:
            raise ValueError('frozen source changed '+path)
    records, traces = {}, {}
    for arm in protocol['arms']:
        stem='building_9002__atlanta_4__'+arm
        record=json.loads((ROOT/'jobs'/(stem+'.result.json')).read_text())
        trace=json.loads((ROOT/'jobs'/(stem+'.result.thermal.json')).read_text())
        if trace['fingerprint']!=digest(ROOT/'protocol.json') or len(trace['rows'])!=2016:
            raise ValueError('thermal trace provenance/length mismatch')
        previous=json.loads((Path('experiments/runs/offhours_hold_short/jobs')/(stem+'.result.json')).read_text())
        for key in ('mean_reward','prod_regret','energy_cost','carbon'):
            if not np.isclose(record['result'][key], previous['result'][key], rtol=0, atol=1e-9):
                raise ValueError('read-only observer changed score '+key)
        for trace_field, result_field, weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if not np.isclose(sum(r[trace_field] for r in trace['rows']), weight*record['result'][result_field],rtol=0,atol=1e-9):
                raise ValueError('trace reward conservation failed')
        records[arm],traces[arm]=record,trace
    if [r['epoch'] for r in traces['phase_ctrl']['rows']] != [r['epoch'] for r in traces['hold_refresh6']['rows']]:
        raise ValueError('unmatched reward intervals')
    legacy_morning=morning_comparison(records['phase_ctrl'],records['hold_refresh6'])
    config=Path('smart_buildings/holdout_scenarios/dev/building_9002/atlanta_4/config.gin').read_text()
    start=pd.Timestamp(ast.literal_eval(re.search(r'^start_timestamp\s*=\s*(.+)$',config,re.M)[1]))
    naive_clock=start.tz is None
    expected_first=(start+pd.Timedelta(minutes=5)).timestamp()
    if traces['phase_ctrl']['rows'][0]['epoch']!=expected_first:
        raise ValueError('simulator start clock does not match frozen config')
    corrected={arm:dict(morning_comfort=corrected_mornings(trace,naive_clock)) for arm,trace in traces.items()}
    morning=morning_comparison(corrected['phase_ctrl'],corrected['hold_refresh6'])
    days={}
    for day, window in corrected['phase_ctrl']['morning_comfort'].items():
        start=window['first_occupied_epoch'];daily={}
        for arm, trace in traces.items():
            daily[arm]={}
            for label,left,right in (('prior_3_to_2h',-10800,-7200),('prior_2_to_1h',-7200,-3600),
                                     ('prior_hour',-3600,0),('first_hour',0,3600),('second_hour',3600,7200),
                                     ('morning',0,7200)):
                daily[arm][label]=summarize([r for r in trace['rows'] if left<=r['epoch']-start<right])
        days[day]=daily
    result=dict(status='complete_verified',morning=morning,legacy_misgrouped_morning=legacy_morning,
                naive_simulator_clock=naive_clock,corrected_windows={a:r['morning_comfort'] for a,r in corrected.items()},days=days,
                means={arm:terms(row) for arm,row in records.items()},
                scope='Atlanta reused 7-day prefix. Legacy UTC conversion misgrouped naive local simulation times; corrected windows use actual simulator dates. Degree-person diagnostics are not the nonlinear comfort score.')
    write_json(ROOT/'verified_review.json',result)
    lines=['# Atlanta morning recovery diagnosis','',result['scope'],'',
           '| Date | Arm | Morning comfort | Cold degree-person-steps | Hot degree-person-steps | Prior-hour temperature K |',
           '|---|---|---:|---:|---:|---:|']
    for day,daily in days.items():
        for arm,row in daily.items():
            m=row['morning'];prior=row['prior_hour'].get('mean_temperature',float('nan'))
            lines.append(f"| {day} | {arm} | {m['comfort_mean']:.8f} | {m['occupied_cold_degree_person_steps']:.4f} | {m['occupied_hot_degree_person_steps']:.4f} | {prior:.4f} |")
    lines += ['','```json',json.dumps(result,indent=2),'```']
    Path('reports/MORNING_RECOVERY_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
