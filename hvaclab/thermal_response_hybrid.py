"""Frozen hybrid forecasting architecture evaluated on a separate trace."""
import argparse
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.policies.thermal_response_model import StateThermalPredictor
from hvaclab.thermal_response_study import examples,forecasts,score,response_measurements


def hybrid(records):
    output=[]
    for row in records:
        r=dict(row)
        if r['state'] not in ('off_entry','on_entry'):r['state_model']=r['linear']
        output.append(r)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.output);path=Path(args.trace);run=json.loads(path.read_text());n=run['steps']
    protocol=dict(trace=str(path),trace_sha256=digest(path),fit_end=n//2,calibration_end=3*n//4,
        architecture='two-step state model at off/on entry; constant velocity at steady commands',
        source_sha256={p:digest(p) for p in ('hvaclab/thermal_response_hybrid.py','hvaclab/thermal_response_study.py','hvaclab/policies/thermal_response_model.py')},
        scope='architecture fixed after Atlanta development; separate weather diagnostic, same building; no reward claim')
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('inputs changed')
    write_json(root/'protocol.json',protocol)
    model=StateThermalPredictor(centre=sum(run['comfort_band'])/2).fit(examples(run,n//2))
    cal=hybrid(list(forecasts(run,model,n//2,3*n//4)));radii={}
    for state in ('off_entry','off_steady','on_entry','on_steady'):
        residuals=sorted(abs(r['state_model']-r['target']) for r in cal if r['state']==state)
        if len(residuals)>=20:radii[state]=residuals[min(len(residuals)-1,int(np.ceil((len(residuals)+1)*.95))-1)]
    test=hybrid(list(forecasts(run,model,3*n//4,n)));scores=score(test,radii,run['comfort_band'])
    off=scores['off_entry'];overall=scores['all'];interval=off.get('interval',{})
    passes=bool(off['n']>=20 and overall['state_model']['mae']<overall['linear']['mae']
        and off['state_model']['mae']<off['linear']['mae'] and off['state_model']['p95']<off['linear']['p95']
        and interval.get('coverage') is not None and interval['coverage']>=.95
        and interval['safe']>0 and interval['false_safe']==0)
    result=dict(protocol=protocol,scores=scores,response_kelvin_per_5min=response_measurements(run),
                passes_diagnostic_gate=passes,controller_changed=False,notebook_exported=False)
    write_json(root/'model.json',dict(coefficients=model.parameters(),radii=radii))
    write_json(root/'results.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
