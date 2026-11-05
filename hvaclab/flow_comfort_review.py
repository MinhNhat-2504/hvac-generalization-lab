"""Attribute full-cohort comfort changes and diagnose matched thermal divergence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hvaclab.coverage import digest,write_json
from hvaclab.offhours_hold_worker import verify_record

FULL=Path('experiments/runs/flow_equivalent_full')
ROOT=Path('experiments/runs/flow_comfort_diagnosis')


def weighted_discomfort(zones):
    """Reconstruct the shipped 5K-knee reward, with hot/cold attribution."""
    occupants=sum(z['occupancy'] for z in zones.values())
    result=dict(hot=0.,cold=0.)
    if occupants<=0:return result
    for z in zones.values():
        for direction,deviation in (('hot',max(0.,z['temperature']-z['high'])),
                                    ('cold',max(0.,z['low']-z['temperature']))):
            shortfall=deviation/5 if deviation<=5 else 1+((deviation-5)/5)**2
            if deviation>0:result[direction]+=.6*z['occupancy']/occupants*shortfall
    return result


def cohort_ledger():
    protocol=json.loads((FULL/'protocol.json').read_text())
    if digest('experiments/runs/damper_refinement/rollouts.json')!=protocol['baseline_sha256']:
        raise ValueError('baseline cache changed')
    rows=[]
    for base in protocol['baseline_rows']:
        b,w=base['building'],base['weather'];split='training' if b.startswith('building_train_') else 'dev'
        path=FULL/'jobs'/f'{split}__{b}__{w}__flow_reset.result.json'
        candidate=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        verify_record(candidate,job)
        trace_path=path.with_suffix('.thermal.json');trace=json.loads(trace_path.read_text())
        if trace['fingerprint']!=digest(FULL/'protocol.json') or len(trace['rows'])!=8640:
            raise ValueError('incomplete or stale candidate trace')
        counts={k:0. for k in ('hot','cold')};by_phase={}
        for row in trace['rows']:
            parts=weighted_discomfort(row['zones'])
            # Proto reward responses use float32; tolerate serialization only.
            if abs(sum(parts.values())-row['comfort'])>2e-7:
                raise ValueError('thermal-to-reward reconstruction mismatch')
            phase=by_phase.setdefault(row['phase'],dict(steps=0,comfort=0.,energy_carbon=0.))
            phase['steps']+=1;phase['comfort']+=row['comfort']/8640
            phase['energy_carbon']+=(row['energy']+row['carbon'])/8640
            for key in counts:counts[key]+=parts[key]/8640
        r=candidate['result'];baseline=-.6*base['prod_regret']/8640;comfort=-.6*r['prod_regret']/8640
        rows.append(dict(split=split,building=b,weather=w,baseline_comfort=baseline,candidate_comfort=comfort,
            comfort_delta=comfort-baseline,loss_delta=abs(r['mean_reward'])-abs(base['mean_reward']),
            candidate_hot_comfort=counts['hot'],candidate_cold_comfort=counts['cold'],candidate_phase=by_phase,
            trace_sha256=digest(trace_path),record_sha256=digest(path)))
    summary={}
    for split,n in (('training',30),('dev',20)):
        selected=[r for r in rows if r['split']==split]
        if len(selected)!=n:raise ValueError('incomplete cohort ledger')
        summary[split]=dict(cases=n,positive=sum(r['comfort_delta']>1e-10 for r in selected),
            negative=sum(r['comfort_delta']<-1e-10 for r in selected),
            gross_regression_mean=sum(max(0.,r['comfort_delta']) for r in selected)/n,
            gross_improvement_mean=sum(min(0.,r['comfort_delta']) for r in selected)/n,
            net_delta_mean=sum(r['comfort_delta'] for r in selected)/n,
            total_loss_delta_mean=sum(r['loss_delta'] for r in selected)/n)
    return dict(status='complete_50_case_ledger',summary=summary,rows=sorted(rows,key=lambda r:r['comfort_delta'],reverse=True),
                scope='Complete cohort reward deltas; hot/cold and phase levels are candidate-only until matched baseline traces exist.')


def compare_traces(control,candidate,clock_naive):
    left,right=control['rows'],candidate['rows']
    if control['schema']!=candidate['schema'] or len(left)!=8640 or len(right)!=8640:
        raise ValueError('unmatched thermal schema/length')
    first={};by_day={};zone_deltas={};events=[];sum_delta=0.
    for i,(a,b) in enumerate(zip(left,right)):
        if a['epoch']!=b['epoch'] or set(a['zones'])!=set(b['zones']):raise ValueError('unaligned intervals or zones')
        for zone in a['zones']:
            if any(a['zones'][zone][f]!=b['zones'][zone][f] for f in ('occupancy','low','high')):
                raise ValueError('different comfort exposures')
        error=max(abs(x-y) for x,y in zip(a['observed_temperature'],b['observed_temperature']))
        metrics={'observed_temperature_0.01K':error>.01,'observed_temperature_0.1K':error>.1,
                 'observed_temperature_0.5K':error>.5,'ahu_run_mismatch':a['run']!=b['run'],
                 'hold_memory_mismatch':a['hold_left']!=b['hold_left'],
                 'sat_command_0.001':max(abs(x-y) for x,y in zip(a['sat'],b['sat']))>.001,
                 'comfort_difference_0.001':abs(a['comfort']-b['comfort'])>.001}
        clock=pd.Timestamp(a['epoch'],unit='s') if clock_naive else pd.Timestamp(a['epoch'],unit='s',tz='UTC').tz_convert(control['schema']['time_zone'])
        for label,trigger in metrics.items():
            if trigger and label not in first:
                first[label]=dict(step=i,clock=str(clock),max_temperature_difference=error,
                                  baseline_phase=a['phase'],candidate_phase=b['phase'],
                                  baseline_run=a['run'],candidate_run=b['run'],baseline_hold=a['hold_left'],candidate_hold=b['hold_left'])
                events.append(dict(label=label,window=[dict(step=j,epoch=left[j]['epoch'],baseline_run=left[j]['run'],candidate_run=right[j]['run'],
                    baseline_sat=left[j]['sat'],candidate_sat=right[j]['sat'],
                    max_temperature_difference=max(abs(x-y) for x,y in zip(left[j]['observed_temperature'],right[j]['observed_temperature'])))
                    for j in range(max(0,i-3),min(len(left),i+4))]))
        day=by_day.setdefault(str(clock.date()),dict(steps=0,baseline_comfort=0.,candidate_comfort=0.,
            delta_hot=0.,delta_cold=0.,run_mismatch_steps=0,max_temperature_difference=0.))
        day['steps']+=1;day['baseline_comfort']+=a['comfort']/8640;day['candidate_comfort']+=b['comfort']/8640
        day['run_mismatch_steps']+=int(a['run']!=b['run']);day['max_temperature_difference']=max(day['max_temperature_difference'],error)
        pa,pb=weighted_discomfort(a['zones']),weighted_discomfort(b['zones'])
        for key in ('hot','cold'):day['delta_'+key]+=(pb[key]-pa[key])/8640
        for zone in a['zones']:
            occupancy=sum(z['occupancy'] for z in a['zones'].values())
            if occupancy>0:
                def contribution(z):
                    d=max(0.,z['temperature']-z['high'],z['low']-z['temperature'])
                    return .6*z['occupancy']/occupancy*(d/5 if d<=5 else 1+((d-5)/5)**2)
                zone_deltas[zone]=zone_deltas.get(zone,0.)+(contribution(b['zones'][zone])-contribution(a['zones'][zone]))/8640
        sum_delta+=(b['comfort']-a['comfort'])/8640
    if abs(sum(zone_deltas.values())-sum_delta)>2e-7:raise ValueError('zone attribution does not conserve reward delta')
    return dict(first_events=first,days=by_day,zone_deltas=sorted(zone_deltas.items(),key=lambda item:item[1],reverse=True),
                comfort_delta=sum_delta,event_windows=events,
                caveat='First divergence and temporal ordering are associations. Pointwise airflow agreement is against the same-state parent request, not the independent V10 trajectory.')


def main(ledger_only=False):
    ROOT.mkdir(parents=True,exist_ok=True);ledger=cohort_ledger();write_json(ROOT/'cohort_ledger.json',ledger)
    if ledger_only:
        print(json.dumps(ledger['summary'],indent=2));return
    p=json.loads((ROOT/'protocol.json').read_text())
    for path,sha in p['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('diagnosis source changed '+path)
    cases={}
    for b,w in p['pairs']:
        control_path=ROOT/'jobs'/f'{b}__{w}__phase_ctrl.result.json'
        control=json.loads(control_path.read_text());job=json.loads(control_path.with_name(control_path.name.replace('.result.json','.job.json')).read_text())
        verify_record(control,job)
        split='training' if b.startswith('building_train_') else 'dev'
        candidate_path=FULL/'jobs'/f'{split}__{b}__{w}__flow_reset.result.json';candidate=json.loads(candidate_path.read_text())
        a=json.loads(control_path.with_suffix('.thermal.json').read_text());c=json.loads(candidate_path.with_suffix('.thermal.json').read_text())
        if a['fingerprint']!=digest(ROOT/'protocol.json') or c['fingerprint']!=digest(FULL/'protocol.json'):
            raise ValueError('matched trace provenance mismatch')
        comparison=compare_traces(a,c,control['clock_semantics']['simulation_timestamp_naive'])
        expected=-.6*(candidate['result']['prod_regret']-control['result']['prod_regret'])/8640
        if abs(expected-comparison['comfort_delta'])>1e-9:raise ValueError('matched diagnostic delta changed')
        comparison['control_record_sha256']=digest(control_path);comparison['candidate_record_sha256']=digest(candidate_path)
        cases[b+'/'+w]=comparison
    probe_path=ROOT/'openloop_probe.json';probe=json.loads(probe_path.read_text())
    if not probe['deterministic_command_replay_exact'] or not probe['baseline_prefix_reproduced']:
        raise ValueError('unverified open-loop replay')
    if digest('hvaclab/flow_openloop_probe.py')!=probe['source_sha256']:
        raise ValueError('open-loop diagnostic source changed')
    for field,path in (
        ('baseline_reference_trace_sha256',ROOT/'jobs/building_train_7__phoenix_3__phase_ctrl.result.thermal.json'),
        ('candidate_trace_sha256',FULL/'jobs/training__building_train_7__phoenix_3__flow_reset.result.thermal.json')):
        if digest(path)!=probe[field]:raise ValueError('open-loop reference changed')
    result=dict(status='complete_verified',cohort=ledger,matched_cases=cases,notebook_exported=False,
                openloop_probe=probe,openloop_probe_sha256=digest(probe_path),
                protocol_sha256=digest(ROOT/'protocol.json'),analysis_sha256=digest(__file__))
    write_json(ROOT/'verified_review.json',result)
    lines=['# Flow-equivalent comfort regression diagnosis','',
           'All50 existing full candidate episodes are accounted for. Two matched V10 controls target the largest positive comfort delta in training and dev. No new policy is promoted.','',
           '| Case | Comfort delta | First0.1K divergence | First AHU run mismatch |','|---|---:|---|---|']
    for case,r in cases.items():
        first=r['first_events'];lines.append(f"| {case} | {r['comfort_delta']:+.9f} | {first.get('observed_temperature_0.1K',{}).get('clock','none')} | {first.get('ahu_run_mismatch',{}).get('clock','none')} |")
    lines+=['','```json',json.dumps(result,indent=2),'```']
    Path('reports/FLOW_COMFORT_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=ledger['summary'],cases={k:{a:b for a,b in v.items() if a not in ('days','event_windows')} for k,v in cases.items()}),indent=2))


if __name__=='__main__':
    import sys
    main('--ledger-only' in sys.argv)
