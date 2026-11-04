"""Frozen occupancy/state selector tested on a new recorded weather trace."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import write_json,digest
from hvaclab.actuator_response_study import PublicActuatorView,records,masked
from hvaclab.policies.actuator_thermal_model import ActuatorThermalPredictor
from hvaclab.policies.adaptive_thermal_forecast import AdaptiveThermalSelector,CANDIDATES


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();directory=Path(args.directory);root=Path(args.output);path=directory/'public_actuators.json';payload=json.loads(path.read_text());n=payload['steps']
    if digest(directory/'damper12_mid.json')!=payload['trace_sha256']:raise ValueError('paired trace changed')
    protocol=dict(public_sha256=digest(path),fit_end=n//2,calibration_end=3*n//4,
        minimum_group_samples=100,minimum_model_gain=.05,
        selection='minimum calibration MAE per fan-state x occupancy; complex models need5% gain over best simple baseline; no test-driven switching',
        source_sha256={p:digest(p) for p in ('hvaclab/adaptive_actuator_study.py','hvaclab/policies/adaptive_thermal_forecast.py','hvaclab/actuator_response_study.py','hvaclab/policies/actuator_thermal_model.py')},
        gate='overall MAE below best aged/linear comparator, occupied MAE and p95 no worse than best comparator, occupied empirical interval coverage>=95%',
        scope='architecture developed after Denver/Atlanta; first separate-weather check Fairbanks; same building, not independent buildings or reward validation')
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('frozen inputs changed')
    write_json(root/'protocol.json',protocol)
    view=PublicActuatorView(payload);train=list(records(view,2,n//2));cal=list(records(view,n//2,3*n//4));test=list(records(view,3*n//4,n))
    for variant in CANDIDATES[2:]:
        model=ActuatorThermalPredictor().fit((r['zone'],r['state'],masked(r['features'],variant),r['target']-r['temperature']) for r in train)
        for row in cal+test:
            increment=model.predict_delta(row['zone'],row['state'],masked(row['features'],variant))
            if increment is None:raise ValueError('untrained fan state')
            row[variant]=row['temperature']+increment
    for row in cal+test:row['aged_sensor']=row['temperature'];row['linear']=row['temperature']+row['slope']
    selector=AdaptiveThermalSelector().fit(cal)
    for row in test:row['adaptive']=selector.predict(row)
    scores={}
    for group in ('all','occupied','empty'):
        rs=[r for r in test if group=='all' or (r['occupancy']>0)==(group=='occupied')]
        values={'n':len(rs)}
        for variant in CANDIDATES+('adaptive',):
            e=[abs((r['adaptive']['temperature'] if variant=='adaptive' else r[variant])-r['target']) for r in rs]
            values[variant]=dict(mae=float(np.mean(e)),p95=float(np.quantile(e,.95))) if e else None
        covered=[r['adaptive']['interval'][0]<=r['target']<=r['adaptive']['interval'][1] for r in rs if r['adaptive']['interval'] is not None]
        values['coverage']=float(np.mean(covered)) if covered else None;values['unknown_intervals']=len(rs)-len(covered)
        scores[group]=values
    a=scores['all'];o=scores['occupied'];passes=bool(o['n']>0 and a['adaptive']['mae']<min(a[v]['mae'] for v in CANDIDATES[:2])
        and o['adaptive']['mae']<=min(o[v]['mae'] for v in CANDIDATES[:2])
        and o['adaptive']['p95']<=min(o[v]['p95'] for v in CANDIDATES[:2]) and o['unknown_intervals']==0
        and o['coverage'] is not None and o['coverage']>=.95)
    result=dict(weather=payload['weather'],protocol=protocol,scores=scores,selection=selector.selected,
        calibration_counts=selector.counts,passes_prediction_gate=passes,controller_changed=False,notebook_exported=False)
    write_json(root/'results.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
