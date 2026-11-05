"""Four-arm AHU damping ablation using the existing physical flow observer."""
from hvaclab.flow_equivalent_worker import main as run_observed


def register():
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['ahu_damped']=('hvaclab.policies.ahu_damped','AHUDampedPolicy')
    for key,damping,flow in (('phase_ctrl',False,False),('flow_reset',False,True),
                             ('ahu_smooth',True,False),('flow_smooth',True,True)):
        _VARIANTS[key]=('ahu_damped',{**_VARIANTS['damper12_mid'][1],
            'offhours_hold_steps':None,'flow_equivalent':flow,'ahu_damping':damping,
            'pressure_floor_pa':200.,'damper_floor':.1,'target_opening':.98,
            'ahu_alpha':.5,'ahu_max_correction_k':.15,'ahu_urgent_k':1.})


if __name__=='__main__':run_observed(register_extra=register)
