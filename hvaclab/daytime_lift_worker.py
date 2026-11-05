"""Physical-audited morning recovery ablation using public schedule inputs."""
from hvaclab.cooling_lift_worker import main as run_observed


def register():
    from hvaclab.guarded_lift_worker import register as parent
    from hvaclab.registry import _OURS,_VARIANTS
    parent()
    _OURS['daytime_lift']=('hvaclab.policies.daytime_cooling_lift','DaytimeCoolingLiftPolicy')
    for key,hours in (('day_lift2',2.),('day_lift4',4.)):
        _VARIANTS[key]=('daytime_lift',{**_VARIANTS['project_lift'][1],
                                     'recovery_hours':hours,'comfort_guard':False})


if __name__=='__main__':run_observed(register_extra=register)
