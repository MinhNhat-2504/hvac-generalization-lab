"""Verify paired plant telemetry; distinguish run-duration and load differences."""
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.plant_runtime_worker import verify_record
from hvaclab.plant_runtime_telemetry import aggregate

ROOT=Path('experiments/runs/plant_runtime_short')


def compare_pair(control,feedback):
    base=control['plant_telemetry'];candidate=feedback['plant_telemetry']
    rows=base['series'];others=candidate['series']
    if len(rows)!=len(others):raise ValueError('unequal series')
    metrics=('ahu_handler_hours','parent_request_handler_hours','hold_handler_hours','heat_override_handler_hours',
             'armed_hold_handler_hours','parent_cooling_handler_hours','parent_warm_handler_hours','pump_hours',
             'pump_without_valve_hours','held_reheat_handler_hours','held_reheat_without_pump_handler_hours','ahu_starts','pump_starts','hold_refreshes','new_hold_episodes')
    phases={}
    for phase in base['phase_totals']:
        a=base['phase_totals'][phase];b=candidate['phase_totals'][phase]
        phases[phase]=dict(control=a,feedback=b,delta={k:b[k]-a[k] for k in metrics},
                           meter_delta_Wh={k:b['metered_Wh'][k]-a['metered_Wh'][k] for k in a['metered_Wh']})
    aligned=[];extra=[0]*len(base['ahu_ids']);removed=extra.copy();reason=dict(parent=0,hold=0,heat_override=0);hold_origins={p:0 for p in ('prestart','occupied','scheduled_empty','offhours')};parent_modes=dict(cooling=0,warm_or_neutral=0)
    by_command={name:dict(steps=0,weighted_energy_carbon_delta=0.,meter_delta_Wh={k:0. for k in ('blower','cooling','pump','gas','heatpump')}) for name in ('identical_run_commands','changed_run_commands')}
    for a,b in zip(rows,others):
        if a['step']!=b['step'] or a['phase']!=b['phase']:raise ValueError('phase/time alignment mismatch')
        if a['phase']!='offhours':continue
        aligned.append((a,b));key='identical_run_commands' if a['run']==b['run'] and a['pump']==b['pump'] else 'changed_run_commands'
        bucket=by_command[key];bucket['steps']+=1
        bucket['weighted_energy_carbon_delta']+=b['weighted_energy']+b['weighted_carbon']-a['weighted_energy']-a['weighted_carbon']
        for k in bucket['meter_delta_Wh']:bucket['meter_delta_Wh'][k]+=(b['watts'][k]-a['watts'][k])*a['dt_seconds']/3600.
        for j,(x,y) in enumerate(zip(a['run'],b['run'])):
            if y and not x:
                extra[j]+=1
                reason['parent']+=b['parent_run'][j];reason['hold']+=int(b['hold'][j]);reason['heat_override']+=int(b['heat_override'][j])
                if b['hold'][j] and b['hold_origin'][j] in hold_origins:hold_origins[b['hold_origin'][j]]+=1
                if b['parent_run'][j]:parent_modes['cooling' if b['parent_sat'][j]<0 else 'warm_or_neutral']+=1
            if x and not y:removed[j]+=1
    for bucket in by_command.values():bucket['weighted_energy_carbon_delta']/=control['result']['steps']
    return dict(ahu_ids=base['ahu_ids'],phase_comparison=phases,offhours_extra_on_handler_steps=extra,
                offhours_removed_on_handler_steps=removed,offhours_extra_on_reason=reason,
                offhours_extra_on_hold_origin=hold_origins,offhours_extra_on_parent_mode=parent_modes,
                offhours_delta_by_command_state=by_command,
                control_transitions=base['transitions'],feedback_transitions=candidate['transitions'])


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json');records={}
    for path,sha in protocol['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('source changed: '+path)
    for path in (ROOT/'jobs').glob('*.result.json'):
        r=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        if r['fingerprint']!=fingerprint:raise ValueError('protocol mismatch')
        verify_record(r,job);task=r['task'];key=(task['building'],task['weather'],task['policy'])
        if key in records:raise ValueError('duplicate record')
        records[key]=r
        for row in r['plant_telemetry']['series']:
            if row['run']!=row['actual_run'] or row['pump']!=row['actual_pump']:raise ValueError('actual state mismatch')
        for phase,b in r['phase_sums'].items():
            phase_rows=[row for row in r['plant_telemetry']['series'] if row['phase']==phase]
            for series_key,bucket_key in (('weighted_energy','energy_cost'),('weighted_carbon','carbon')):
                if not np.isclose(sum(row[series_key] for row in phase_rows),.2*b[bucket_key],atol=1e-8,rtol=1e-10):raise ValueError('metered scorer response drift')
    if set(records)!={(b,w,a) for b,w in protocol['pairs'] for a in protocol['arms']}:raise ValueError('incomplete paired cohort')
    cases={f'{b}/{w}':compare_pair(records[b,w,'phase_ctrl'],records[b,w,'phase_all']) for b,w in protocol['pairs']}
    result=dict(status='complete_verified',cases=cases,exact_v10_parity_steps=sum(records[b,w,'phase_ctrl']['shadow_action_parity_steps'] for b,w in protocol['pairs']),
        original_reward_and_phase_terms_reproduced=True,controller_changed=False,hold_changed=False,notebook_exported=False,
        scope='read-only selected2016-step prefix diagnosis, no remedy or hidden-score claim')
    write_json(ROOT/'verified_review.json',result)
    lines=['# AHU/pump/cooling-hold runtime diagnosis ? 2026-11-05','',
        'All four paired2016-step records matched prior rewards and phase terms. Actual device run states matched commands. AHU runtime is summed machine-hours across units; pump hours are single-system hours. Hold override means a pre-hold OFF request was changed to ON, not merely armed hold memory.','',
        '| Case / offhours | V10 | Full feedback | Delta |','|---|---:|---:|---:|']
    for name,case in cases.items():
        off=case['phase_comparison']['offhours']
        for metric in ('ahu_handler_hours','parent_request_handler_hours','hold_handler_hours','pump_hours','ahu_starts','pump_starts'):
            lines.append(f"| {name} / {metric} | {off['control'][metric]:.5f} | {off['feedback'][metric]:.5f} | {off['delta'][metric]:+.5f} |")
        for meter,value in off['meter_delta_Wh'].items():lines.append(f"| {name} / {meter}_Wh | {off['control']['metered_Wh'][meter]:.3f} | {off['feedback']['metered_Wh'][meter]:.3f} | {value:+.3f} |")
    lines+=['','```json',json.dumps(result,indent=2),'```','',
        'The data identify runtime/reason/power associations on matched trajectories. Price/carbon weighting and building normalizers mean kWh changes are not direct shares of the competition score. A hold override does not prove it is avoidable without comfort loss. Identical run commands with different power can reflect airflow, valve, temperature and thermal-history differences. No policy or hold rule was changed.']
    Path('reports/PLANT_RUNTIME_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,case in cases.items():
        off=case['phase_comparison']['offhours'];print(name,'OFFHOURS',json.dumps(dict(delta=off['delta'],meter_delta_Wh=off['meter_delta_Wh'],extra_reason=case['offhours_extra_on_reason'],by_command=case['offhours_delta_by_command_state'])))


if __name__=='__main__':main()
