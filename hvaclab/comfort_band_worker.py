"""Fixed factorial comparison of centre PI, band PI and flow allocation."""
from hvaclab.flow_equivalent_worker import main as run_observed


def register():
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['comfort_band_pi']=('hvaclab.policies.comfort_band_pi','ComfortBandPIPolicy')
    for key,band,flow in (('phase_ctrl',False,False),('flow_reset',False,True),
                         ('band_pi',True,False),('band_flow',True,True)):
        _VARIANTS[key]=('comfort_band_pi',{**_VARIANTS['damper12_mid'][1],
            'offhours_hold_steps':None,'flow_equivalent':flow,'band_pi':band,'comfort_reserve_k':.5,
            'pressure_floor_pa':200.,'damper_floor':.1,'target_opening':.98})


if __name__=='__main__':run_observed(register_extra=register)
