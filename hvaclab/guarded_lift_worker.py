"""Compare signed allocation projection and comfort restoration separately."""
from hvaclab.cooling_lift_worker import main as run_observed


def register():
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['guarded_lift']=('hvaclab.policies.guarded_cooling_lift','GuardedCoolingLiftPolicy')
    for key,lift,flow,projection,guard in (('phase_ctrl',False,False,False,False),
        ('flow_lift',True,True,False,False),('project_lift',True,True,True,False),('guard_lift',True,True,True,True)):
        _VARIANTS[key]=('guarded_lift',{**_VARIANTS['damper12_mid'][1],
            'offhours_hold_steps':None,'flow_equivalent':flow,'cooling_lift':lift,
            'project_lift':projection,'comfort_guard':guard,'projection_limit_k':.5,
            'projection_steps':2.,'guard_margin_k':.25,'supply_reduction_k':2.,
            'pressure_floor_pa':200.,'damper_floor':.1,'target_opening':.98})


if __name__=='__main__':run_observed(register_extra=register)
