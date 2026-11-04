"""Reproduce forecast identification summaries and occupancy-stratified diagnostics."""
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import write_json,digest
from hvaclab.policies.thermal_response_model import StateThermalPredictor
from hvaclab.thermal_response_study import examples,forecasts,response_measurements
from hvaclab.sensor_restart_review import metrics


def now_metrics(run):
    model=StateThermalPredictor(centre=sum(run['comfort_band'])/2).fit(examples(run,run['steps']//2))
    forecasts_=list(forecasts(run,model,3*run['steps']//4,run['steps']))
    result={}
    for group in ('all','occupied','empty'):
        rows=[r for r in forecasts_ if group=='all' or (run['trace'][r['step']]['occupancy']>0)==(group=='occupied')]
        values=dict(n=len(rows))
        for key in ('now_model','now_linear','aged_sensor'):
            error=[abs((run['trace'][r['step']]['observed_before'][r['zone']] if key=='aged_sensor' else r[key])-r['target_now']) for r in rows]
            values[key]=dict(mae=float(np.mean(error)),p95=float(np.quantile(error,.95))) if error else None
        result[group]=values
    return result


def main():
    root=Path('experiments/runs');result={}
    for name,path in (('atlanta',root/'sensor_restart_short/sensor_restart.json'),('denver',root/'thermal_response_denver/sensor_restart.json')):
        run=json.loads(path.read_text());report=json.loads((root/('thermal_response_hybrid_'+name)/'results.json').read_text())
        for source,sha in report['protocol']['source_sha256'].items():
            if digest(source)!=sha:raise ValueError('source changed: '+source)
        if digest(path)!=report['protocol']['trace_sha256']:raise ValueError('trace changed')
        result[name]=dict(trace=str(path),hybrid=report,now=now_metrics(run))
    baseline=root/'thermal_response_denver/damper12_mid.json'
    result['denver_v10']=dict(prefix=metrics(baseline),response=response_measurements(json.loads(baseline.read_text())))
    result['denver_sensor_restart']=metrics(root/'thermal_response_denver/sensor_restart.json')
    result['decision']='retain diagnostic predictor; reject current coasting integration; collect continuous actuator and weather features before next model'
    write_json(root/'thermal_response_identification/final_review.json',result)
    lines=['# State-conditioned forecast review ? 2026-11-04','',
           'First 50% fit, next25% calibration, last25% evaluation. Atlanta used for architecture development; Denver is another weather trace in the same building. Each model fits only its own early sensor data. No unseen-building or leaderboard improvement claim.','',
           '| Trace | Overall MAE hybrid / linear / persistence (K) | Off-entry MAE hybrid / linear (K) | Off-entry radius (K) | Diagnostic gate |',
           '|---|---|---|---|---|']
    for name in ('atlanta','denver'):
        d=result[name]['hybrid'];a=d['scores']['all'];o=d['scores']['off_entry']
        lines.append(f"| {name} | {a['state_model']['mae']:.5f} / {a['linear']['mae']:.5f} / {a['persistence']['mae']:.5f} | {o['state_model']['mae']:.5f} / {o['linear']['mae']:.5f} | {o['interval']['calibration_radius']:.5f} | {d['passes_diagnostic_gate']} |")
    lines+=['','Denver persistence beats the hybrid overall. Off-entry empirical radius1.838K exceeds the1.5K half-band and permits zero interval-safe off entries; no controller integration. Point-prediction gains alone are insufficient.',
            '','Exploratory current-temperature metrics (K), not the prespecified gate:','', '```json',json.dumps({n:result[n]['now'] for n in ('atlanta','denver')},indent=2),'```','',
            'Current-temperature gains are concentrated in empty periods. In occupied Denver steps, model MAE .20083 vs aged sensor .17868, despite lower p95 .53446 vs .83678. Do not replace occupied PI feedback on this evidence.',
            '','Descriptive responses are conditioned on actual operational histories, not randomized interventions. Commands other than AHU on/off, occupancy and weather differ between groups. Next identification must record SAT, pressure, zone damper/reheat actions and outdoor temperature. Preserve V10 control. No full sweep or notebook generated.']
    Path('reports/STATE_FORECAST_REVIEW_2026_11_04.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('saved final_review.json and local review')


if __name__=='__main__':main()
