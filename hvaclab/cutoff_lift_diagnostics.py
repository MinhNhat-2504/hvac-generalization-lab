"""Pair internal PI/hold histories without interpreting them as thermal truth."""
import json


def summarize(root,records):
    result={}
    for b,w in sorted({(b,w) for b,w,_ in records}):
        traces={arm:json.loads((root/'jobs'/f'{b}__{w}__{arm}.result.controller.json').read_text()) for arm in
            ('phase_ctrl','project_lift','day_lift2','cut_lift14','cut_lift16')}
        physical=json.loads((root/'jobs'/f'{b}__{w}__phase_ctrl.result.thermal.json').read_text())
        case={}
        for arm in ('project_lift','day_lift2','cut_lift14','cut_lift16'):
            case[arm]={}
            for reference in ('phase_ctrl','day_lift2'):
                phases={}
                for a,c,t in zip(traces[reference]['rows'],traces[arm]['rows'],physical['rows']):
                    label=t['phase'];v=phases.setdefault(label,dict(steps=0,integral_signed_delta=0.,integral_absolute_difference=0.,
                        pi_output_absolute_difference=0.,hold_mean_delta=0.,hold_increases_reference=0,hold_increases_candidate=0,
                        holds_started_reference=0,holds_started_candidate=0))
                    zones=traces[arm]['schema']['zone_ids'];v['steps']+=1
                    v['integral_signed_delta']+=sum(c['integral_after'].get(z,0)-a['integral_after'].get(z,0) for z in zones)/len(zones)
                    v['integral_absolute_difference']+=sum(abs(c['integral_after'].get(z,0)-a['integral_after'].get(z,0)) for z in zones)/len(zones)
                    v['pi_output_absolute_difference']+=sum(abs(c['zone_u'].get(z,0)-a['zone_u'].get(z,0)) for z in zones)/len(zones)
                    v['hold_mean_delta']+=sum(y-x for x,y in zip(a['hold_after'],c['hold_after']))/len(a['hold_after'])
                    for name,row in (('reference',a),('candidate',c)):
                        v['hold_increases_'+name]+=sum(y>x for x,y in zip(row['hold_before'],row['hold_after']))
                        v['holds_started_'+name]+=row['holds_started_delta']
                for v in phases.values():
                    for key in ('integral_signed_delta','integral_absolute_difference','pi_output_absolute_difference','hold_mean_delta'):v[key]/=v['steps']
                case[arm][reference]=phases
        result[b+'/'+w]=case
    return result
