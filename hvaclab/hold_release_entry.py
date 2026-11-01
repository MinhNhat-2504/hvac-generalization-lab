"""Isolated registry and packaging adapter; never rewrites running study files."""
import ast
from pathlib import Path
import sys

ARMS={'release_ctrl':dict(release_enabled=False,cold_deficit_k=.5,confirmation_steps=3),
      'release_cold':dict(release_enabled=True,cold_deficit_k=.5,confirmation_steps=3),
      'release_early':dict(release_enabled=True,cold_deficit_k=.25,confirmation_steps=3),
      'release_slow':dict(release_enabled=True,cold_deficit_k=.5,confirmation_steps=6)}


def register():
    from hvaclab.registry import _OURS,_VARIANTS
    _OURS['comfort_hold_release']=('hvaclab.policies.comfort_hold_release','ComfortHoldReleasePolicy')
    base=dict(_VARIANTS['valve8_gentle'][1])
    for key,kwargs in ARMS.items():
        _VARIANTS[key]=('comfort_hold_release',{**base,**kwargs})


def assemble(key):
    from hvaclab.submission.experimental import assemble_variant
    from hvaclab.registry import _VARIANTS
    if key not in ARMS:raise ValueError(key)
    base,_=assemble_variant('valve8_gentle')
    base=base[:base.rindex('\nEQUIVALENT_TO =')]
    path=Path(__file__).parent/'policies/comfort_hold_release.py'
    source=path.read_text(encoding='utf-8')
    lines=source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('hvaclab.'):
            for i in range(node.lineno-1,node.end_lineno):lines[i]=''
    params=dict(_VARIANTS[key][1])
    source=(base+'\n'+''.join(lines)+f'\nEQUIVALENT_TO = {key!r}\n'
            'class FinalPolicy(ComfortHoldReleasePolicy):\n'
            '    def __init__(self,building_meta_data,action_spec):\n'
            f'        super().__init__(building_meta_data,action_spec,**{params!r})\n')
    compile(source,'release_policy.py','exec')
    return source,params


def package():
    from hvaclab.submission import experimental as bundle
    original=bundle.assemble_variant
    def adapter(key):
        # assemble() calls the unmodified assembler for the V7 parent.
        bundle.assemble_variant=original
        try:return assemble(key)
        finally:bundle.assemble_variant=adapter
    sources=bundle.variant_sources('valve8_gentle')+('hvaclab/policies/comfort_hold_release.py',)
    bundle.SUPPORTED+=tuple(ARMS)
    bundle.REHEAT_CURVE+=tuple(ARMS)  # runtime marker only; source adapter below
    bundle.variant_sources=lambda key:sources
    bundle.assemble_variant=adapter
    sys.argv=[sys.argv[0]]+[x for x in sys.argv[1:] if x!='--package']
    bundle.main()


def main():
    register()
    if '--package' in sys.argv:package()
    else:
        from hvaclab.coverage import main as job_main
        job_main()


if __name__=='__main__':main()
