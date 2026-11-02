"""Isolated single-worker valve interaction on the submitted V9 controller."""
import sys
from hvaclab import pipeline_entry as base

SPECS={'parallel_valve':dict(control='parallelv_ctrl',base_key='damper11_less',
    control_run='heating_damper',heating_class=True,
    arms={'parallelv_ctrl':dict(reheat_exponent=.75),
          'parallelv_more':dict(reheat_exponent=.625),
          'parallelv_less':dict(reheat_exponent=.875),
          'parallelv_linear':dict(reheat_exponent=1.)},
    dependencies=('hold_release_calibration','curve_refinement','heating_damper','damper_refinement'),
    title='Single-worker valve characteristic interaction on submitted V9')}


def register():
    base.register()
    from hvaclab.registry import _VARIANTS
    params=dict(_VARIANTS['damper11_less'][1])
    for key,kwargs in SPECS['parallel_valve']['arms'].items():
        _VARIANTS[key]=('heating_damper_curve',{**params,**kwargs})


def assemble(key):
    previous=base.SPECS
    base.SPECS={**previous,**SPECS}
    try:return base.assemble(key)
    finally:base.SPECS=previous


def package():
    previous=base.SPECS
    base.SPECS={**previous,**SPECS}
    try:base.package()
    finally:base.SPECS=previous


def main():
    register()
    if '--package' in sys.argv:package()
    else:
        from hvaclab.coverage import main as worker
        worker()


if __name__=='__main__':main()
