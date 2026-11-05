"""Attribute completed projection-screen comfort changes by morning and sign."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.flow_comfort_review import weighted_discomfort

ROOT=Path('experiments/runs/guarded_lift_short')


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for name,sha in p['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('study source changed: '+name)
    cases={};inputs={}
    for b,w in p['pairs']:
        records={};traces={}
        for arm in ('phase_ctrl','project_lift','guard_lift'):
            path=ROOT/'jobs'/f'{b}__{w}__{arm}.result.json'
            records[arm]=json.loads(path.read_text());tp=path.with_suffix('.thermal.json')
            traces[arm]=json.loads(tp.read_text());inputs[str(path)]=digest(path);inputs[str(tp)]=digest(tp)
            if traces[arm]['fingerprint']!=fingerprint or len(traces[arm]['rows'])!=p['steps']:
                raise ValueError('incomplete or stale trace')
        starts=[v['first_occupied_epoch'] for v in records['phase_ctrl']['morning_comfort'].values()]
        result={}
        for arm in ('project_lift','guard_lift'):
            sums={k:dict(hot_delta=0.,cold_delta=0.,steps=0) for k in ('morning','rest')}
            for x,y in zip(traces['phase_ctrl']['rows'],traces[arm]['rows']):
                if x['epoch']!=y['epoch'] or set(x['zones'])!=set(y['zones']):raise ValueError('unpaired trace')
                if any(x['zones'][z]['occupancy']!=y['zones'][z]['occupancy'] for z in x['zones']):
                    raise ValueError('changed occupancy exposure')
                before,after=weighted_discomfort(x['zones']),weighted_discomfort(y['zones'])
                if max(abs(sum(before.values())-x['comfort']),abs(sum(after.values())-y['comfort']))>2e-7:
                    raise ValueError('reward reconstruction failed')
                label='morning' if any(0<=x['epoch']-s<7200 for s in starts) else 'rest'
                for sign in ('hot','cold'):sums[label][sign+'_delta']+=(after[sign]-before[sign])/p['steps']
                sums[label]['steps']+=1
            result[arm]=sums
        cases[b+'/'+w]=result
    report=dict(status='complete_verified',protocol_sha256=fingerprint,inputs_sha256=inputs,cases=cases,
        scope='Paired contributions normalized by4032 steps. Morning is first two actual occupied hours; offline attribution only, no simulator truth enters control. Not a causal estimate of a schedule gate.')
    write_json(ROOT/'morning_attribution.json',report)
    Path('reports/GUARDED_LIFT_MORNING_ATTRIBUTION.md').write_text('# Projection morning attribution\n\n'+report['scope']+'\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
