"""Reproduce a diagnostic forecast bundle and summarize actuator identification."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
from hvaclab.coverage import write_json,digest
from hvaclab.actuator_response_study import PublicActuatorView,records,masked
from hvaclab.policies.actuator_thermal_model import ActuatorThermalPredictor
from hvaclab.policies.adaptive_thermal_forecast import AdaptiveThermalSelector,CANDIDATES


def main():
    root=Path('experiments/runs/actuator_identification')
    d=json.loads((root/'fairbanks_adaptive/results.json').read_text())
    for source,sha in d['protocol']['source_sha256'].items():
        if digest(source)!=sha:raise ValueError('source changed: '+source)
    data=json.loads((root/'fairbanks_3/public_actuators.json').read_text())
    if digest(root/'fairbanks_3/public_actuators.json')!=d['protocol']['public_sha256']:raise ValueError('public trace changed')
    view=PublicActuatorView(data);train=list(records(view,2,data['steps']//2));cal=list(records(view,data['steps']//2,3*data['steps']//4));models={}
    for name in CANDIDATES[2:]:
        model=ActuatorThermalPredictor().fit((r['zone'],r['state'],masked(r['features'],name),r['target']-r['temperature']) for r in train)
        models[name]=model.parameters()
        for row in cal:
            prediction=model.predict_delta(row['zone'],row['state'],masked(row['features'],name))
            if prediction is None:raise ValueError('missing calibration state')
            row[name]=row['temperature']+prediction
    for row in cal:row['aged_sensor']=row['temperature'];row['linear']=row['temperature']+row['slope']
    selector=AdaptiveThermalSelector().fit(cal)
    if selector.selected!=d['selection']:raise ValueError('calibrated selection did not reproduce')
    write_json(root/'fairbanks_adaptive/diagnostic_bundle.json',dict(models=models,selected=selector.selected,
        radii=selector.radii,schema=data['schema'],public_sha256=d['protocol']['public_sha256'],
        scope='same-building diagnostic bundle, not submission; fit and calibration segments only'))
    with Path('reports/ACTUATOR_FORECAST_REVIEW_2026_11_04.md').open('a',encoding='utf-8') as report:
        report.write('\n## Fixed-rule Fairbanks check\n\n```json\n'+json.dumps(d,indent=2)+'\n```\n')
        report.write('\nThe gate passed for current-temperature prediction only, not closed-loop reward. '
            'Models and selector were fit/calibrated using only the early75% of this new trace. '
            'The same building is reused; new-building transfer is untested. '
            'Residual coverage is empirical, after forecast selection on the calibration segment. '
            'The next bounded check is a causal online estimator on additional buildings, with V10 unchanged as a control. '
            'No static per-scenario coefficient file should be treated as a general competition policy.\n')
    print('saved diagnostic_bundle.json and final review')


if __name__=='__main__':main()
