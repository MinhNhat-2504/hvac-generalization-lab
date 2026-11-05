"""Package only after full mean, physical and all50 morning validation."""
import ast
import json
from pathlib import Path
import sys
from hvaclab.coverage import digest,write_json

KEY='flow_equivalent_full'


def assemble():
    from hvaclab.pipeline_entry import assemble as assemble_parent,register
    register()
    source,params=assemble_parent('damper12_mid')
    source=source[:source.rindex('\nEQUIVALENT_TO =')]
    for path in ('hvaclab/policies/offhours_hold.py','hvaclab/policies/flow_equivalent.py'):
        text=Path(path).read_text(encoding='utf-8');lines=text.splitlines(keepends=True)
        for node in ast.parse(text).body:
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('hvaclab.'):
                for i in range(node.lineno-1,node.end_lineno):lines[i]=''
        source+='\n'+''.join(lines)
    params=dict(params,offhours_hold_steps=None,flow_equivalent=True,pressure_floor_pa=200.,damper_floor=.1,target_opening=.98)
    source+=f'\nEQUIVALENT_TO = {KEY!r}\nclass FinalPolicy(FlowEquivalentPolicy):\n    def __init__(self,building_meta_data,action_spec):\n        super().__init__(building_meta_data,action_spec,**{params!r})\n'
    compile(source,KEY,'exec');return source,params


def verify(source,key,steps=576):
    import numpy as np
    from smart_buildings import evaluate as ev
    from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
    namespace={};exec(compile(source,KEY,'exec'),namespace);_,params=assemble();pairs=0;changed=0
    for directory,b,w in (('training_scenarios','building_train_1','fairbanks_1'),
                           ('holdout_scenarios/dev','building_9002','singapore_2'),
                           ('holdout_scenarios/dev','building_9000','phoenix_2')):
        scenario=Path('smart_buildings')/directory/b/w;env=ev.load_hybrid_environment(scenario/'config.gin')
        meta=ev.create_building_meta_data(env,scenario);p=namespace['FinalPolicy'](meta,env.action_spec());q=FlowEquivalentPolicy(meta,env.action_spec(),**params)
        obs,_=env.reset()
        for _ in range(steps):
            a,c=p.get_action(obs),q.get_action(obs)
            for field in a:np.testing.assert_array_equal(a[field],c[field])
            obs,_,terminated,truncated,_=env.step(ev.validate_action(a,env.action_spec()));pairs+=1
            if terminated or truncated:break
        changed+=p.rebalanced_handler_steps
    if changed<=0:raise ValueError('standalone test did not activate pressure reset')
    return dict(action_pairs=pairs,changed_handler_steps=changed)


def main():
    from hvaclab.submission import experimental as bundle
    root=Path('experiments/runs/flow_equivalent_full');review=json.loads((root/'verified_review.json').read_text());p=json.loads((root/'protocol.json').read_text())
    if not review['eligible_for_packaging'] or review['protocol_sha256']!=digest(root/'protocol.json'):
        raise ValueError('full validation does not permit packaging')
    for path,sha in p['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('source changed before packaging '+path)
    source,params=assemble();parity=verify(source,KEY)
    sources=bundle.variant_sources('persist_432')+('hvaclab/policies/heating_damper_curve.py','hvaclab/policies/offhours_hold.py','hvaclab/policies/flow_equivalent.py')
    bundle.SUPPORTED+=(KEY,);bundle.REHEAT_CURVE+=(KEY,)
    bundle.variant_sources=lambda key:sources
    bundle.assemble_variant=lambda key:(source,params)
    old=sys.argv
    try:sys.argv=[old[0],'--variant',KEY];bundle.main()
    finally:sys.argv=old
    out=Path('experiments/submission')/KEY
    manifest=json.loads((out/'manifest.json').read_text());manifest['action_parity_verified']=True;manifest['parity']=parity;write_json(out/'manifest.json',manifest)
    write_json(out/'validation.json',dict(protocol_sha256=review['protocol_sha256'],review_sha256=digest(root/'verified_review.json'),
        full_local_statistics=review['statistics'],parity=parity,notebook_sha256=digest(out/'submission.ipynb'),organizer_score_unknown=True))
    (out/'SUBMISSION_DESCRIPTION.txt').write_text('V10 zone PI and36h hold with flow-equivalent fan pressure reset. Pressure and every damper are coordinated to preserve per-zone airflow. Complete local training/dev and standalone parity verified; no hidden-score claim.',encoding='utf-8')


if __name__=='__main__':main()
