"""Read-only paired afternoon/night carryover attribution, never policy feedback."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hvaclab.coverage import digest,write_json
from hvaclab.flow_comfort_review import weighted_discomfort
from hvaclab.morning_recovery_review import corrected_mornings

ROOT=Path('experiments/runs/daytime_lift_short')
OUT=Path('experiments/runs/thermal_carryover_analysis')


def local_clock(epoch,naive,zone):
    return pd.Timestamp(epoch,unit='s') if naive else pd.Timestamp(epoch,unit='s',tz='UTC').tz_convert(zone)


def windows(first_epoch,naive,zone,start_hour):
    day=local_clock(first_epoch,naive,zone).normalize()
    prior=day-pd.DateOffset(days=1)
    values=(('prior_afternoon',prior+pd.Timedelta(hours=14),prior+pd.Timedelta(hours=19)),
            ('prior_evening',prior+pd.Timedelta(hours=19),day),
            ('overnight',day,day+pd.Timedelta(hours=start_hour-1)),
            ('prestart',day+pd.Timedelta(hours=start_hour-1),local_clock(first_epoch,naive,zone)),
            ('morning',local_clock(first_epoch,naive,zone),local_clock(first_epoch,naive,zone)+pd.Timedelta(hours=2)))
    return {name:(left.timestamp(),right.timestamp()) for name,left,right in values}


def paired_summary(base,candidate,left,right):
    selected=[(a,b) for a,b in zip(base,candidate) if left<=a['epoch']<right]
    expected=round((right-left)/300)
    if not selected:return dict(steps=0,expected_steps=expected,complete=False)
    if any(a['epoch']!=b['epoch'] or set(a['zones'])!=set(b['zones']) for a,b in selected):
        raise ValueError('unpaired intervals')
    actual=np.array([[b['zones'][z]['temperature']-a['zones'][z]['temperature'] for z in sorted(a['zones'])] for a,b in selected])
    public=np.array([[y-x for x,y in zip(a['observed_temperature'],b['observed_temperature'])] for a,b in selected])
    hot=cold=0.
    for a,b in selected:
        pa,pb=weighted_discomfort(a['zones']),weighted_discomfort(b['zones'])
        hot+=pb['hot']-pa['hot'];cold+=pb['cold']-pa['cold']
    n=len(selected)
    def vector_delta(key):return sum(sum(y-x for x,y in zip(a[key],b[key])) for a,b in selected)
    return dict(steps=n,expected_steps=expected,complete=n==expected,
        mean_temperature_delta_K=float(actual.mean()),end_temperature_delta_K=float(actual[-1].mean()),
        max_abs_zone_temperature_delta_K=float(np.abs(actual).max()),
        public_temperature_delta_K=float(public.mean()),
        positive_temperature_exposure_Kh=float(np.maximum(actual,0).sum()/actual.shape[1]/12),
        hot_comfort_delta=hot/n,cold_comfort_delta=cold/n,
        energy_carbon_sum_delta=sum(b['energy']+b['carbon']-a['energy']-a['carbon'] for a,b in selected),
        ahu_machine_hours_delta=vector_delta('run')/12,
        pump_hours_delta=sum(b['pump']-a['pump'] for a,b in selected)/12,
        mean_zone_flow_sum_delta=sum(sum(z['flow'] for z in b['zones'].values())-sum(z['flow'] for z in a['zones'].values()) for a,b in selected)/n,
        mean_hold_steps_delta=vector_delta('hold_left')/n/len(selected[0][0]['hold_left']),
        run_mismatch_fraction=sum(a['run']!=b['run'] for a,b in selected)/n,
        mean_normalized_sat_delta=vector_delta('sat')/n/len(selected[0][0]['sat']),
        mean_normalized_damper_delta=vector_delta('damper')/n/len(selected[0][0]['damper']))


def morning_rooms(base,candidate,first_epoch,boundaries=None):
    pairs=[(a,b) for a,b in zip(base,candidate) if first_epoch<=a['epoch']<first_epoch+7200]
    rooms={z:0. for z in pairs[0][0]['zones']} if pairs else {}
    for a,b in pairs:
        total=sum(v['occupancy'] for v in a['zones'].values())
        if total<=0:continue
        for z in rooms:
            def contribution(row):
                v=row['zones'][z];d=max(0.,v['temperature']-v['high'])
                return .6*v['occupancy']/total*(d/5 if d<=5 else 1+((d-5)/5)**2)
            rooms[z]+=(contribution(b)-contribution(a))/len(pairs)
    result=[]
    for z in sorted(rooms,key=rooms.get,reverse=True)[:3]:
        item=dict(room=z,morning_hot_delta=rooms[z],boundary_temperature_delta_K={})
        for label,epoch in (boundaries or {'arrival':first_epoch}).items():
            before=[(a,b) for a,b in zip(base,candidate) if a['epoch']<epoch]
            item['boundary_temperature_delta_K'][label]=None if not before else before[-1][1]['zones'][z]['temperature']-before[-1][0]['zones'][z]['temperature']
        result.append(item)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for name,sha in protocol['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('frozen source changed '+name)
    review=json.loads((ROOT/'verified_review.json').read_text());cases={};inputs={}
    groups={a:{g:[] for g in ('failed','passed')} for a in ('day_lift2','day_lift4')}
    for b,w in protocol['pairs']:
        traces={};records={}
        for arm in ('phase_ctrl','day_lift2','day_lift4'):
            path=ROOT/'jobs'/f'{b}__{w}__{arm}.result.json';tp=path.with_suffix('.thermal.json')
            records[arm]=json.loads(path.read_text());traces[arm]=json.loads(tp.read_text())
            inputs[path.as_posix()]=digest(path);inputs[tp.as_posix()]=digest(tp)
            t=traces[arm];r=records[arm]
            if t['fingerprint']!=fingerprint or r['fingerprint']!=fingerprint or len(t['rows'])!=4032:
                raise ValueError('stale trace')
            if any(y['epoch']-x['epoch']!=300 for x,y in zip(t['rows'],t['rows'][1:])):raise ValueError('nonuniform trace')
            mornings=corrected_mornings(t,r['clock_semantics']['simulation_timestamp_naive'])
            if set(mornings)!=set(r['morning_comfort']):raise ValueError('morning dates mismatch')
            for day,values in mornings.items():
                for key,value in values.items():
                    other=r['morning_comfort'][day][key]
                    if isinstance(value,str):
                        if value!=other:raise ValueError('morning clock mismatch')
                    elif abs(value-other)>1e-8:raise ValueError('morning reconstruction failed')
            for field,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
                if abs(sum(x[field] for x in t['rows'])-weight*r['result'][total])>1e-8:raise ValueError('reward conservation failed')
            for row in t['rows']:
                if abs(sum(weighted_discomfort(row['zones']).values())-row['comfort'])>2e-7:raise ValueError('comfort reconstruction failed')
        base=traces['phase_ctrl'];days={}
        for arm in ('day_lift2','day_lift4'):
            candidate=traces[arm]
            if base['schema']!=candidate['schema']:raise ValueError('schema mismatch')
            for x,y in zip(base['rows'],candidate['rows']):
                if x['epoch']!=y['epoch'] or set(x['zones'])!=set(y['zones']):raise ValueError('alignment mismatch')
                if any(x['zones'][z]['occupancy']!=y['zones'][z]['occupancy'] for z in x['zones']):raise ValueError('occupancy mismatch')
            failed={d['day'] for d in review['cases'][arm][b+'/'+w]['morning_vs_v10']['failed_mornings']}
            rows={}
            for day,m in records['phase_ctrl']['morning_comfort'].items():
                spans=windows(m['first_occupied_epoch'],records['phase_ctrl']['clock_semantics']['simulation_timestamp_naive'],base['schema']['time_zone'],base['schema']['morning_start_hour'])
                data={label:paired_summary(base['rows'],candidate['rows'],*bounds) for label,bounds in spans.items()}
                complete=all(v['complete'] for v in data.values())
                row=dict(failed_morning=day in failed,full_prior_history=complete,windows=data,
                    worst_hot_rooms=morning_rooms(base['rows'],candidate['rows'],m['first_occupied_epoch'],
                        dict(prior_19h=spans['prior_evening'][0],prestart=spans['prestart'][0],arrival=m['first_occupied_epoch'])))
                rows[day]=row
                if complete:groups[arm]['failed' if day in failed else 'passed'].append(row)
            days[arm]=rows
        cases[b+'/'+w]=days
    aggregates={}
    for arm,by_group in groups.items():
        aggregates[arm]={}
        for group,rows in by_group.items():
            summary={label:{key:float(np.mean([row['windows'][label][key] for row in rows])) for key in
                ('mean_temperature_delta_K','end_temperature_delta_K','positive_temperature_exposure_Kh','ahu_machine_hours_delta','pump_hours_delta','mean_hold_steps_delta','mean_zone_flow_sum_delta','hot_comfort_delta','cold_comfort_delta','run_mismatch_fraction')} for label in next(iter(rows))['windows']} if rows else {}
            aggregates[arm][group]=dict(days=len(rows),windows=summary)
    room_counts={}
    for arm in ('day_lift2','day_lift4'):
        failed=[row for arms in cases.values() for row in arms[arm].values() if row['failed_morning']]
        room_counts[arm]=dict(failed_mornings=len(failed),top_hot_room_warmer_at_prestart=sum(row['worst_hot_rooms'][0]['boundary_temperature_delta_K']['prestart']>0 for row in failed),
            top_hot_room_warmer_at_arrival=sum(row['worst_hot_rooms'][0]['boundary_temperature_delta_K']['arrival']>0 for row in failed))
    result=dict(status='complete_verified',protocol_sha256=fingerprint,analysis_sha256=digest(__file__),inputs_sha256=inputs,cases=cases,aggregates=aggregates,room_counts=room_counts,
        scope='Read-only four reused14-day prefixes. Post-step thermal states, matched public histories and normalized commands. Calendar afternoon14-19/evening19-24, midnight to metadata prestart, then actual occupancy+2h. Group means weight days equally and mix cases, descriptive only. Positive air-temperature exposure is not stored thermal energy; wall temperatures and true cooling power unavailable. Same controller law on different history does not mean equal commands. No causal claim, no notebook.')
    write_json(OUT/'verified_analysis.json',result)
    lines=['# Afternoon and night carryover diagnosis','',result['scope'],'',
        '| Arm | Morning group | Days | Window | Mean air delta K | End air delta K | AHU hours delta | Hot comfort delta |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for arm,by_group in aggregates.items():
        for group,summary in by_group.items():
            for label,v in summary['windows'].items():lines.append(f"| {arm} | {group} | {summary['days']} | {label} | {v['mean_temperature_delta_K']:+.5f} | {v['end_temperature_delta_K']:+.5f} | {v['ahu_machine_hours_delta']:+.4f} | {v['hot_comfort_delta']:+.8f} |")
    lines+=['','The failed-morning groups show less aggregate overnight AHU runtime and warmer air on average, but individual exceptions rule out a universal runtime-only explanation. In the worst Singapore morning, total AHU runtime is unchanged overnight while the dominant hot room is already warmer before prestart. Phoenix has a failing morning where its dominant room starts colder at prestart and becomes warmer by arrival. Whole-building mean temperature and total runtime are insufficient comfort guards.',
        '', 'Next proposed causal screen: isolate late-afternoon allocation (cutoff14h versus16h), preserve V10 after cutoff and before recovery, and record per-zone PI integral plus per-handler hold transitions. Compare with V10 and the unrestricted projected arm; preserve all prior acceptance gates. This analysis launches no rollout and provides no evidence that a cutoff will succeed.',
        '', '```json',json.dumps(result,indent=2),'```']
    Path('reports/THERMAL_CARRYOVER_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
