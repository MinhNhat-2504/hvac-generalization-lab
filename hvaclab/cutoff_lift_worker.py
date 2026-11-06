"""Read-only PI/hold observer around the physically audited cooling worker."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import json
from pathlib import Path
import sys
import numpy as np
from hvaclab.coverage import digest,write_json


def register():
    from hvaclab.daytime_lift_worker import register as parent
    from hvaclab.registry import _OURS,_VARIANTS
    parent()
    _OURS['cutoff_lift']=('hvaclab.policies.cutoff_cooling_lift','CutoffCoolingLiftPolicy')
    for key,hour in (('cut_lift14',14.),('cut_lift16',16.)):
        _VARIANTS[key]=('cutoff_lift',{**_VARIANTS['day_lift2'][1],'cutoff_hour':hour})


def observe_action(policy,action,rows):
    """Copies only controller/public state; never reads or alters simulator state."""
    def capture(obs):
        before=dict(policy._zone_integral);hold=list(policy._hold_left)
        started=policy.holds_started
        command=action(obs)
        allowed=policy._allocation_window(obs) if hasattr(policy,'_allocation_window') else None
        if allowed is False:
            for key in command:
                if not np.array_equal(command[key],policy.last_parent_action[key]):
                    raise ValueError('disabled allocation differs from same-step V10')
            if policy.last_allocations:raise ValueError('disabled-window allocation retained')
        rows.append(dict(step=len(rows),public_hour=int(policy._get_hour_and_day(obs)[0]),
            allocation_allowed=allowed,pi_active=bool(policy._pi_active),
            integral_before=before,integral_after=dict(policy._zone_integral),
            zone_u=dict(policy._zone_u),hold_before=hold,hold_after=list(policy._hold_left),
            holds_started_delta=policy.holds_started-started,
            retained_allocations=sorted(policy.last_allocations)))
        return command
    return capture


def main():
    from smart_buildings import evaluate as ev
    from hvaclab.cooling_lift_worker import main as run
    initialize=ev.initialize_policy;rows=[];schema={}
    def observed(cls,meta,spec):
        policy=initialize(cls,meta,spec)
        schema.update(zone_ids=[v for c in policy.ahu_controllers for v,_,_ in c['zones']],
            ahu_ids=[c['ahu_id'] for c in policy.ahu_controllers],
            integral_limit=policy.zone_integral_limit,recovery_hours=getattr(policy,'recovery_hours',None),
            cutoff_hour=getattr(policy,'cutoff_hour',None))
        policy.get_action=observe_action(policy,policy.get_action,rows)
        return policy
    ev.initialize_policy=observed
    try:run(register_extra=register)
    finally:ev.initialize_policy=initialize
    job=json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text());path=Path(job['result_path'])
    record=json.loads(path.read_text())
    if len(rows)!=record['result']['steps']:raise ValueError('incomplete controller trace')
    sidecar=path.with_suffix('.controller.json')
    write_json(sidecar,dict(fingerprint=job['fingerprint'],task=job['task'],schema=schema,rows=rows,
        semantics='Read-only before/after internal state. Hold increases/decreases are net transitions, not all refresh events. Same-step parent parity checked whenever allocation disabled.'))
    record['controller_audit']=dict(steps=len(rows),sidecar_sha256=digest(sidecar),
        disabled_steps=sum(x['allocation_allowed'] is False for x in rows))
    write_json(path,record)
    print('PI_AND_HOLD_TRACE_VERIFIED',len(rows),flush=True)


if __name__=='__main__':main()
