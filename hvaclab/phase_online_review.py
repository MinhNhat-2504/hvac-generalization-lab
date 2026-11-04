"""Verify phase ablations and report additive cost booking versus causal effects."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.phase_online_worker import verify_record
from hvaclab.policies.phase_online_actuator import PHASES

ROOT=Path('experiments/runs/phase_online_short')


def terms(record,phase=None):
    r=record['result'] if phase is None else record['phase_sums'][phase]
    n=record['result']['steps']
    return dict(loss=-r['total_reward']/n if phase is None else -r['reward_sum']/n,
                comfort=-.6*r['prod_regret']/n,energy=.2*r['energy_cost']/n,carbon=.2*r['carbon']/n)


def main():
    p=json.loads((ROOT/'protocol.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    for source,sha in p['source_sha256'].items():
        if digest(source)!=sha:raise ValueError('source changed: '+source)
    records={}
    for path in (ROOT/'jobs').glob('*.result.json'):
        record=json.loads(path.read_text());job=json.loads(path.with_name(path.name.replace('.result.json','.job.json')).read_text())
        if job['fingerprint']!=fingerprint:raise ValueError('wrong protocol')
        verify_record(record,job);t=record['task'];key=(t['building'],t['weather'],t['policy'])
        if key in records:raise ValueError('duplicate record')
        records[key]=record
    expected={(b,w,a) for b,w in p['pairs'] for a in p['arms']}
    if set(records)!=expected:raise ValueError('incomplete cohort')
    overall={};phase_mean={};diagnostics={}
    for arm in p['arms']:
        rows=[records[b,w,arm] for b,w in p['pairs']]
        overall[arm]={k:sum(terms(r)[k] for r in rows)/len(rows) for k in ('loss','comfort','energy','carbon')}
        phase_mean[arm]={phase:{k:sum(terms(r,phase)[k] for r in rows)/len(rows) for k in ('loss','comfort','energy','carbon')} for phase in PHASES}
        diagnostics[arm]=[{**r['task'],**r['online_diagnostics'],
                          'phase_steps':{ph:r['phase_sums'][ph]['steps'] for ph in PHASES},
                          'reward_occupied_steps':{ph:r['phase_sums'][ph]['reward_occupied_steps'] for ph in PHASES}} for r in rows]
    delta={a:{phase:{k:phase_mean[a][phase][k]-phase_mean['phase_ctrl'][phase][k] for k in ('loss','comfort','energy','carbon')} for phase in PHASES} for a in p['arms'] if a!='phase_ctrl'}
    gates={}
    for arm in ('phase_prestart','phase_occupied'):
        allowed=arm.removeprefix('phase_')
        rows=[records[b,w,arm] for b,w in p['pairs']]
        for r in rows:
            applied=r['online_diagnostics']['applied_by_phase']
            if any(n for phase,n in applied.items() if phase!=allowed):raise ValueError('feedback outside requested phase')
        active=all(r['online_diagnostics']['applied_by_phase'][allowed]>0 for r in rows)
        gates[arm]=dict(active_in_each_case=active,change_pct=100*(overall[arm]['loss']/overall['phase_ctrl']['loss']-1),
            passes_short_gate=bool(active and overall[arm]['loss']<.99*overall['phase_ctrl']['loss']
                                  and overall[arm]['comfort']<=overall['phase_ctrl']['comfort']))
    residual={k:overall['phase_all'][k]-overall['phase_prestart'][k]-overall['phase_occupied'][k]+overall['phase_ctrl'][k] for k in ('loss','comfort','energy','carbon')}
    cases=[]
    for b,w in p['pairs']:
        cases.append(dict(building=b,weather=w,arms={a:terms(records[b,w,a]) for a in p['arms']}))
    result=dict(status='complete_verified',overall=overall,phase_contributions=phase_mean,delta_vs_control=delta,
                gates=gates,interaction_plus_scheduled_empty_residual=residual,cases=cases,diagnostics=diagnostics,
                exact_shadow_steps=sum(records[b,w,'phase_ctrl']['shadow_action_parity_steps'] for b,w in p['pairs']),
                full_sweep_started=False,notebook_exported=False)
    write_json(ROOT/'verified_review.json',result)
    lines=['# Phase-specific online PI compensation ? 2026-11-05','',
        'All eight2016-step jobs verified against frozen provenance, phase conservation and old control/all-arm totals. Phase contributions use total prefix steps, not separate unweighted phase averages. Decision-time occupancy may differ from reward-time occupancy at boundaries.','',
        '| Arm | Loss | Comfort | Energy | Carbon | Change vs V10 |','|---|---:|---:|---:|---:|---:|']
    for arm,r in overall.items():lines.append(f"| {arm} | {r['loss']:.8f} | {r['comfort']:.8f} | {r['energy']:.8f} | {r['carbon']:.8f} | {100*(r['loss']/overall['phase_ctrl']['loss']-1):+.3f}% |")
    lines+=['','Mean all-arm minus V10 costs booked by decision-time phase:','',
        '| Phase | Loss delta | Comfort delta | Energy delta | Carbon delta |','|---|---:|---:|---:|---:|']
    for phase,r in delta['phase_all'].items():lines.append(f"| {phase} | {r['loss']:+.8f} | {r['comfort']:+.8f} | {r['energy']:+.8f} | {r['carbon']:+.8f} |")
    lines+=['','```json',json.dumps(result,indent=2),'```','',
        'Phase cost booking is not direct causal attribution: thermal mass, integral states and online learning transmit effects across intervals. All-arm treatment also includes scheduled-empty feedback. The reported residual therefore combines interactions and scheduled-empty treatment, not a pure two-factor interaction.',
        '','These are selected previously used dev cases and short episodes, not full validation or hidden Kaggle performance. Do not promote a notebook from this analysis alone.']
    Path('reports/PHASE_ONLINE_SHORT_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics','cases','phase_contributions')},indent=2))


if __name__=='__main__':main()
