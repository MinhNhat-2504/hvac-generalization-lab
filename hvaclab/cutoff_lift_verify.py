"""Verify controller traces and preserve physical/control-prefix checks."""
import json
import math
from pathlib import Path
from hvaclab.coverage import digest
from hvaclab.guarded_lift_verify import verify_guarded


def verify_cutoff(record,job):
    verify_guarded(record,job)
    path=Path(job['result_path']).with_suffix('.controller.json');audit=record['controller_audit']
    if digest(path)!=audit['sidecar_sha256']:raise ValueError('controller trace hash mismatch')
    trace=json.loads(path.read_text());thermal=json.loads(Path(job['result_path']).with_suffix('.thermal.json').read_text())
    if trace['task']!=job['task'] or trace['fingerprint']!=job['fingerprint']:raise ValueError('wrong controller provenance')
    rows=trace['rows'];n=record['result']['steps'];schema=trace['schema']
    if len(rows)!=n or len(thermal['rows'])!=n or audit['steps']!=n:raise ValueError('incomplete controller trace')
    previous=None
    for i,(row,physical) in enumerate(zip(rows,thermal['rows'])):
        if row['step']!=i or row['hold_after']!=physical['hold_left']:raise ValueError('controller/thermal clock mismatch')
        if len(row['hold_before'])!=len(schema['ahu_ids']):raise ValueError('handler count mismatch')
        if previous and (row['hold_before']!=previous['hold_after'] or row['integral_before']!=previous['integral_after']):raise ValueError('controller state discontinuity')
        for field in ('integral_before','integral_after','zone_u'):
            if not set(row[field])<=set(schema['zone_ids']):raise ValueError('unknown zone')
            limit=1. if field=='zone_u' else schema['integral_limit']
            if any(not math.isfinite(v) or abs(v)>limit+1e-8 for v in row[field].values()):raise ValueError('PI state out of bounds')
        if row['allocation_allowed'] is False and row['retained_allocations']:raise ValueError('allocation outside window')
        cutoff=schema['cutoff_hour']
        if cutoff is not None and row['public_hour']>=cutoff and row['allocation_allowed'] is not False:raise ValueError('cutoff ignored')
        previous=row
    if sum(x['allocation_allowed'] is False for x in rows)!=audit['disabled_steps']:raise ValueError('disabled count mismatch')
