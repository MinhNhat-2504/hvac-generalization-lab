"""Fixed chronological thermal-response identification on recorded public sensors."""
import argparse
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.policies.thermal_response_model import StateThermalPredictor,fan_state


def mapping(run):
    zones={}
    for handler,indices in enumerate(run['ahu_zone_indices']):
        for zone in indices:zones[zone]=handler
    if len(zones)!=len(run['zone_ids']):raise ValueError('missing zone assignment')
    return zones


def examples(run,end):
    rows=run['trace'];zones=mapping(run)
    for t in range(2,end-1):
        for zone,handler in zones.items():
            # obs[t+1]-obs[t] is caused by command[t-1], not command[t].
            state=fan_state(rows[t-1]['ahu_on'][handler],rows[t-2]['ahu_on'][handler])
            temp=rows[t]['observed_before'][zone]
            slope=temp-rows[t-1]['observed_before'][zone]
            delta=rows[t+1]['observed_before'][zone]-temp
            yield zone,state,temp,slope,rows[t]['occupancy'],delta


def forecasts(run,model,start,end):
    rows=run['trace'];zones=mapping(run)
    for t in range(max(2,start),end-2):
        for zone,handler in zones.items():
            temp=rows[t]['observed_before'][zone]
            slope=temp-rows[t-1]['observed_before'][zone]
            occupied=rows[t]['occupancy']
            state_now=fan_state(rows[t-1]['ahu_on'][handler],rows[t-2]['ahu_on'][handler])
            state_next=fan_state(rows[t]['ahu_on'][handler],rows[t-1]['ahu_on'][handler])
            first=model.predict_delta(zone,state_now,temp,slope,occupied)
            if first is None:continue
            now=temp+first
            second=model.predict_delta(zone,state_next,now,first,occupied)
            if second is None:continue
            # Targets are later public sensors, not simulator truth used as features.
            yield dict(step=t,zone=zone,state=state_next,
                       target_now=rows[t+1]['observed_before'][zone],
                       target=rows[t+2]['observed_before'][zone],
                       state_model=now+second,linear=temp+2*slope,persistence=temp,
                       now_model=now,now_linear=temp+slope)


def score(records,radii,band):
    result={}
    for state in ('all','off_entry','off_steady','on_entry','on_steady'):
        rows=[r for r in records if state=='all' or r['state']==state]
        if not rows:result[state]=dict(n=0);continue
        values=dict(n=len(rows),origin_steps=len({r['step'] for r in rows}))
        for name in ('state_model','linear','persistence'):
            errors=np.array([abs(r[name]-r['target']) for r in rows])
            values[name]=dict(mae=float(np.mean(errors)),p95=float(np.quantile(errors,.95)),
                false_safe=sum(band[0]<=r[name]<=band[1] and not band[0]<=r['target']<=band[1] for r in rows))
        covered=[];unknown=0;safe=0;false_safe=0
        for r in rows:
            radius=radii.get(r['state'])
            if radius is None:unknown+=1;continue
            covered.append(abs(r['state_model']-r['target'])<=radius)
            if band[0]<=r['state_model']-radius and r['state_model']+radius<=band[1]:
                safe+=1;false_safe+=int(not band[0]<=r['target']<=band[1])
        values['interval']=dict(calibration_radius=radii.get(state),coverage=float(np.mean(covered)) if covered else None,
                                unknown=unknown,safe=safe,false_safe=false_safe)
        result[state]=values
    return result


def response_measurements(run):
    rows=run['trace'];groups={}
    for t in range(1,len(rows)):
        for handler,zones in enumerate(run['ahu_zone_indices']):
            state=fan_state(rows[t]['ahu_on'][handler],rows[t-1]['ahu_on'][handler])
            bucket=groups.setdefault(state,dict(increments=[],handler_steps=0))
            bucket['handler_steps']+=1
            bucket['increments'].extend(rows[t]['physical_after'][z]-rows[t]['physical_before'][z] for z in zones)
    result={}
    for state,bucket in groups.items():
        x=np.array(bucket['increments'])
        result[state]=dict(handler_steps=bucket['handler_steps'],zone_samples=len(x),mean=float(x.mean()),
                           p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',default='experiments/runs/sensor_restart_short/sensor_restart.json')
    parser.add_argument('--output',default='experiments/runs/thermal_response_identification')
    args=parser.parse_args();path=Path(args.trace);run=json.loads(path.read_text())
    n=run['steps'];fit_end=n//2;cal_end=3*n//4;root=Path(args.output)
    protocol=dict(trace=str(path),trace_sha256=digest(path),steps=n,fit_end=fit_end,calibration_end=cal_end,
                  ridge=10.,minimum_zone_samples=8,interval_quantile=.95,
                  source_sha256={p:digest(p) for p in ('hvaclab/thermal_response_study.py','hvaclab/policies/thermal_response_model.py')},
                  gate='2-step MAE below linear overall and at off-entry; off-entry p95 below linear; calibrated off-entry coverage >=95%; at least one interval-safe off-entry; no false-safe interval at off-entry',
                  scope='selected trace temporal diagnostic; no independent generalization or reward claim')
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:
        raise ValueError('frozen inputs changed; choose new output')
    write_json(root/'protocol.json',protocol)
    model=StateThermalPredictor(centre=sum(run['comfort_band'])/2).fit(examples(run,fit_end))
    calibration=list(forecasts(run,model,fit_end,cal_end))
    radii={}
    for state in ('off_entry','off_steady','on_entry','on_steady'):
        residuals=sorted(abs(r['state_model']-r['target']) for r in calibration if r['state']==state)
        if len(residuals)>=20:radii[state]=residuals[min(len(residuals)-1,int(np.ceil((len(residuals)+1)*.95))-1)]
    test=list(forecasts(run,model,cal_end,n));scores=score(test,radii,run['comfort_band'])
    overall=scores['all'];off=scores['off_entry'];interval=off.get('interval',{})
    passes=bool(off['n']>=20 and overall['state_model']['mae']<overall['linear']['mae']
                and off['state_model']['mae']<off['linear']['mae'] and off['state_model']['p95']<off['linear']['p95']
                and interval.get('coverage') is not None and interval['coverage']>=.95
                and interval['safe']>0 and interval['false_safe']==0)
    result=dict(status='complete',protocol=protocol,response_kelvin_per_5min=response_measurements(run),
                calibration_samples=len(calibration),test_samples=len(test),scores=scores,
                passes_diagnostic_gate=passes,controller_changed=False,notebook_exported=False)
    write_json(root/'model.json',dict(coefficients=model.parameters(),radii=radii))
    write_json(root/'results.json',result)
    report=Path('reports/THERMAL_RESPONSE_IDENTIFICATION.md')
    report.write_text('# State-conditioned thermal response identification\n\n'
        'First 50% fit, next25% calibration, last25% evaluation. Delayed public sensor labels only; '
        'private field used solely for descriptive response measurements. '
        'Four actuator states, per-zone ridge coefficients shrunk toward pooled state coefficients. '
        'Quantile intervals are empirical and zone/time samples are dependent; no formal coverage guarantee. '
        'This selected previously analyzed trace is not untouched validation.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
