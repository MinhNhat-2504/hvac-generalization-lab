"""Check physical targets against the explicitly declared cooling reserve."""
from hvaclab.guarded_lift_verify import verify_guarded


def verify_reserve(record,job):
    verify_guarded(record,job)
    diagnostic=record['hold_diagnostics'];arm=job['task']['policy']
    if arm in ('phase_ctrl','project_lift') and diagnostic.get('reserve_zone_steps',0)!=0:
        raise ValueError('control received reserve')
    if diagnostic.get('reserve_zone_steps',0)<0:raise ValueError('invalid reserve count')
    limit={'reserve10':1.1,'reserve20':1.2}.get(arm,1.)
    if record['lift_audit']['max_target_cooling_multiplier']>limit+1e-12:
        raise ValueError('per-arm cooling reserve exceeded')
