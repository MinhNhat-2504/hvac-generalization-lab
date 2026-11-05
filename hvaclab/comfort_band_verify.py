"""Execution and reference checks shared by the comfort-band PI scheduler/reviewer."""
import json
from pathlib import Path
from hvaclab.coverage import digest
from hvaclab.offhours_hold_worker import verify_record
from hvaclab.flow_equivalent_validation import fan_audit_passes


def verify_band(record,job):
    verify_record(record,job)
    audit=record['flow_audit'];diagnostics=record['hold_diagnostics']
    if audit['checked_zone_steps']<=0 or audit['max_relative_actual_flow_error']>2e-6 or not fan_audit_passes(audit):
        raise ValueError('comfort-band PI physical audit failed')
    if diagnostics['max_error_reduction_k']>1.+1e-9:raise ValueError('PI error reduction bound exceeded')
    if job['task']['policy'] in ('phase_ctrl','flow_reset') and diagnostics['band_zone_steps']!=0:
        raise ValueError('control arm used a band error')
    reference=job.get('prior_trace')
    if reference:
        if digest(reference['path'])!=reference['sha256']:raise ValueError('prior control trace changed')
        trace=json.loads(Path(reference['path']).read_text());rows=trace['rows'][:job['task']['steps']]
        if len(rows)!=record['result']['steps']:raise ValueError('prior prefix incomplete')
        for source,total,weight in (('comfort','prod_regret',-.6),('energy','energy_cost',.2),('carbon','carbon',.2)):
            if abs(sum(r[source] for r in rows)-weight*record['result'][total])>1e-8:
                raise ValueError('control did not reproduce prior prefix '+source)
        current=json.loads(Path(job['result_path']).with_suffix('.thermal.json').read_text())
        if len(current['rows'])!=len(rows):raise ValueError('current thermal prefix incomplete')
        for a,b in zip(rows,current['rows']):
            if a['epoch']!=b['epoch'] or a['observed_temperature']!=b['observed_temperature'] or a['run']!=b['run'] or a['sat']!=b['sat']:
                raise ValueError('disabled filter control trajectory changed')
