"""Child-local adapters for the two queued Version 8 follow-up studies."""
import ast
from pathlib import Path
import sys

SPECS={
 'curve_refinement':dict(control='curve10_ctrl',
    arms={'curve10_ctrl':dict(reheat_exponent=.75),
          'curve10_half':dict(reheat_exponent=.5),
          'curve10_mid':dict(reheat_exponent=.625),
          'curve10_light':dict(reheat_exponent=.875)},
    dependencies=('hold_release_calibration',),
    title='Valve refinement on the 36-hour cooling hold'),
 'heating_damper':dict(control='damper11_ctrl',
    arms={'damper11_ctrl':dict(heating_damper_exponent=1.),
          'damper11_more':dict(heating_damper_exponent=.75),
          'damper11_less':dict(heating_damper_exponent=1.25),
          'damper11_soft':dict(heating_damper_exponent=1.5)},
    dependencies=('hold_release_calibration','curve_refinement'),
    title='Heating damper response with the Version 8 reheat curve'),
 'damper_refinement':dict(control='damper12_ctrl',base_key='damper11_less',
    control_run='heating_damper',heating_class=True,
    arms={'damper12_ctrl':dict(heating_damper_exponent=1.25),
          'damper12_light':dict(heating_damper_exponent=1.1),
          'damper12_mid':dict(heating_damper_exponent=1.4),
          'damper12_strong':dict(heating_damper_exponent=1.75)},
    dependencies=('hold_release_calibration','curve_refinement','heating_damper'),
    title='Heating damper exponent refinement on submitted Version 9'),
 'damper_persistence':dict(control='persist13_ctrl',base_key='damper11_less',
    control_run='heating_damper',heating_class=True,
    arms={'persist13_ctrl':dict(hold_steps=432),
          'persist13_day':dict(hold_steps=288),
          'persist13_two':dict(hold_steps=576),
          'persist13_long':dict(hold_steps=720)},
    dependencies=('hold_release_calibration','curve_refinement','heating_damper','damper_refinement'),
    title='Cooling persistence interaction with submitted Version 9 heating airflow')}


def register():
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['heating_damper_curve']=('hvaclab.policies.heating_damper_curve','HeatingDamperCurvePolicy')
    for study,spec in SPECS.items():
        base=dict(_VARIANTS[spec.get('base_key','persist_432')][1])
        for key,params in spec['arms'].items():
            _VARIANTS[key]=('heating_damper_curve' if study=='heating_damper' or spec.get('heating_class') else 'reheat_curve',
                            {**base,**params})


def key_study(key):
    return next(name for name,spec in SPECS.items() if key in spec['arms'])


def assemble(key):
    from hvaclab.submission.experimental import assemble_variant
    from hvaclab.registry import _VARIANTS
    study=key_study(key)
    base,_=assemble_variant('persist_432')
    base=base[:base.rindex('\nEQUIVALENT_TO =')]
    parent='ReheatCurvePolicy'
    if study=='heating_damper' or SPECS[study].get('heating_class'):
        source=(Path(__file__).parent/'policies/heating_damper_curve.py').read_text(encoding='utf-8')
        lines=source.splitlines(keepends=True)
        for node in ast.parse(source).body:
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('hvaclab.'):
                for i in range(node.lineno-1,node.end_lineno):lines[i]=''
        base+='\n'+''.join(lines)
        parent='HeatingDamperCurvePolicy'
    params=dict(_VARIANTS[key][1])
    result=(base+f'\nEQUIVALENT_TO = {key!r}\nclass FinalPolicy({parent}):\n'
            '    def __init__(self,building_meta_data,action_spec):\n'
            f'        super().__init__(building_meta_data,action_spec,**{params!r})\n')
    compile(result,key,'exec')
    return result,params


def package():
    from hvaclab.submission import experimental as bundle
    key=sys.argv[sys.argv.index('--variant')+1]
    assembled=assemble(key)
    sources=bundle.variant_sources('persist_432')
    family=key_study(key)
    if family=='heating_damper' or SPECS[family].get('heating_class'):
        sources+=('hvaclab/policies/heating_damper_curve.py',)
    def adapter(k):
        if k!=key:raise ValueError('unexpected packaging key')
        return assembled
    bundle.SUPPORTED+=tuple(k for spec in SPECS.values() for k in spec['arms'])
    bundle.REHEAT_CURVE+=tuple(k for spec in SPECS.values() for k in spec['arms'])
    # Source list affects both concatenation and manifest. Use the captured
    # base list so no recursive call enters the adapter.
    bundle.variant_sources=lambda k:sources
    bundle.assemble_variant=adapter
    sys.argv=[a for a in sys.argv if a!='--package']
    bundle.main()


def main():
    register()
    if '--package' in sys.argv:package()
    else:
        from hvaclab.coverage import main as worker_main
        worker_main()


if __name__=='__main__':main()
