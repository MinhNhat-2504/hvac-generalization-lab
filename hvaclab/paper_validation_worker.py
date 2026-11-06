"""Frozen full-episode paper comparison; identical physical audit for all arms."""
from hvaclab.reserve_lift_worker import main


def register():
    from hvaclab.registry import _VARIANTS
    _VARIANTS['flow_only']=('guarded_lift',{**_VARIANTS['project_lift'][1],
        'cooling_lift':False,'project_lift':False})


if __name__=='__main__':main(register_extra=register)
