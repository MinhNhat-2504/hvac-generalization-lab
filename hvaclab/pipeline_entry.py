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
    title='Heating damper response with the Version 8 reheat curve')}


def register():
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['heating_damper_curve']=('hvaclab.policies.heating_damper_curve','HeatingDamperCurvePolicy')
    base=dict(_VARIANTS['persist_432'][1])
    for study,spec in SPECS.items():
        for key,params in spec['arms'].items():
            _VARIANTS[key]=('heating_damper_curve' if study=='heating_damper' else 'reheat_curve',
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
    if study=='heating_damper':
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
    if key_study(key)=='heating_damper':sources+=('hvaclab/policies/heating_damper_curve.py',)
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
