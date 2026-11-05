"""Trace the fixed early-PI recovery candidate and its exact V10 shadow."""
from hvaclab import offhours_hold_worker as base
from hvaclab import morning_recovery_worker as trace


def main():
    original=base.register
    def register():
        original()
        from hvaclab.registry import _OURS,_VARIANTS
        _OURS['morning_precool']=('hvaclab.policies.morning_precool','MorningPrecoolPolicy')
        _VARIANTS['precool_early']=('morning_precool',{**_VARIANTS['damper12_mid'][1],
             'offhours_hold_steps':6,'recovery_extra_hours':1.,'recovery_warm_margin':.5,'recovery_cold_margin':.5})
    base.register=register
    try:trace.main()
    finally:base.register=original


if __name__=='__main__':main()
