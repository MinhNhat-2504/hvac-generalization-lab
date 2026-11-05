"""Verify thermal-allocation audits and unchanged centre-PI baseline prefixes."""
import json
from pathlib import Path
from hvaclab.coverage import digest
from hvaclab.offhours_hold_worker import verify_record


def verify_lift(record,job):
    verify_record(record,job);audit=record['lift_audit']
    if audit['max_relative_actual_flow_error']>2e-6 or audit['max_relative_estimated_cooling_error']>2e-6 or audit['supply_guard_violations']!=0:
        raise ValueError('thermal allocation physical/model audit failed')
    if job['task']['policy'] in ('phase_ctrl','flow_reset') and record['hold_diagnostics']['lift_handler_steps']!=0:
        raise ValueError('control arm changed supply/flow allocation')
    reference=job.get('prior_trace')
    if reference:
        if digest(reference['path'])!=reference['sha256']:raise ValueError('prior prefix changed')
        trace=json.loads(Path(reference['path']).read_text());rows=trace['rows'][:job['task']['steps']]
        if len(rows)!=record['result']['steps']:raise ValueError('prior prefix incomplete')
        for source,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(r[source] for r in rows)-weight*record['result'][total])>1e-8:raise ValueError('control prefix scorer mismatch')
        current=json.loads(Path(job['result_path']).with_suffix('.thermal.json').read_text())
        if len(current['rows'])!=len(rows):raise ValueError('current prefix incomplete')
        for a,b in zip(rows,current['rows']):
            if any(a[field]!=b[field] for field in ('epoch','observed_temperature','run','sat','damper','reheat')):
                raise ValueError('control trajectory mismatch')
