"""Child-local coasting adapter and mechanism diagnostics."""
import ast
import json
import os
from pathlib import Path
import sys

SPECS={'thermal_coast':dict(control='coast_ctrl',base_key='damper12_mid',
    control_run='damper_refinement',
    arms={'coast_ctrl':dict(coast_mode='off'),
          'coast_empty':dict(coast_mode='empty',coast_margin_k=.5,forecast_steps=3),
          'coast_reserved':dict(coast_mode='all',coast_margin_k=.75,forecast_steps=3),
          'coast_lookahead':dict(coast_mode='all',coast_margin_k=.5,forecast_steps=6)},
    dependencies=('hold_release_calibration','curve_refinement','heating_damper',
                  'damper_refinement','damper_persistence'),
    title='Structural per-handler thermal coasting on V10')}


def register():
    from hvaclab.pipeline_entry import register as parents
    parents()
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['thermal_coast']=('hvaclab.policies.thermal_coasting','ThermalCoastingPolicy')
    base=dict(_VARIANTS['damper12_mid'][1])
    for key,params in SPECS['thermal_coast']['arms'].items():
        _VARIANTS[key]=('thermal_coast',{**base,'minimum_on_steps':6,**params})


def assemble(key):
    from hvaclab.pipeline_entry import assemble as parent
    from hvaclab.registry import _VARIANTS
    base,_=parent('damper12_mid')
    base=base[:base.rindex('\nEQUIVALENT_TO =')]
    source=(Path(__file__).parent/'policies/thermal_coasting.py').read_text(encoding='utf-8')
    lines=source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('hvaclab.'):
            for i in range(node.lineno-1,node.end_lineno):lines[i]=''
    params=dict(_VARIANTS[key][1])
    source=(base+'\n'+''.join(lines)+f'\nEQUIVALENT_TO = {key!r}\n'
            'class FinalPolicy(ThermalCoastingPolicy):\n'
            '    def __init__(self,building_meta_data,action_spec):\n'
            f'        super().__init__(building_meta_data,action_spec,**{params!r})\n')
    compile(source,key,'exec')
    return source,params


def package():
    from hvaclab.submission import experimental as bundle
    key=sys.argv[sys.argv.index('--variant')+1]
    assembled=assemble(key)
    sources=bundle.variant_sources('persist_432')+('hvaclab/policies/heating_damper_curve.py',
                                                 'hvaclab/policies/thermal_coasting.py')
    bundle.SUPPORTED+=tuple(SPECS['thermal_coast']['arms'])
    bundle.REHEAT_CURVE+=tuple(SPECS['thermal_coast']['arms'])
    bundle.assemble_variant=lambda k:assembled
    bundle.variant_sources=lambda k:sources
    sys.argv=[a for a in sys.argv if a!='--package']
    bundle.main()


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    register()
    if '--package' in sys.argv:
        package()
        return
    # Capture the policy object only in this offline worker, so diagnostics
    # never introduce simulator/reward access in the submitted controller.
    from smart_buildings import evaluate as ev
    original=ev.initialize_policy
    captured=[]
    def initialize(*args,**kwargs):
        policy=original(*args,**kwargs)
        captured.append(policy)
        return policy
    ev.initialize_policy=initialize
    try:
        from hvaclab.coverage import main as worker
        worker()
    finally:ev.initialize_policy=original
    if '--job' in sys.argv and captured:
        from hvaclab.coverage import write_json
        descriptor=Path(sys.argv[sys.argv.index('--job')+1])
        job=json.loads(descriptor.read_text())
        path=Path(job['result_path'])
        record=json.loads(path.read_text())
        record['thermal_diagnostics']={name:getattr(captured[-1],name) for name in
            ('coast_handler_steps','coast_occupied_handler_steps','coast_entries','guard_restarts')}
        write_json(path,record)


if __name__=='__main__':main()
