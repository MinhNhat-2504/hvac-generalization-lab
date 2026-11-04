"""Summarize the fixed short diagnostic rejection screen without exporting a policy."""
import json
from pathlib import Path
from hvaclab.coverage import digest,write_json
from hvaclab.thermal_diagnostics import analyze_probe,compare_probes


def metrics(path):
    run=json.loads(Path(path).read_text());low,high=run['comfort_band']
    return dict(policy=run['policy'],steps=run['steps'],
                loss=-sum(row['reward'] for row in run['trace'])/run['steps'],
                occupied_zone_steps_outside=sum(not low<=x<=high for row in run['trace']
                    if row['occupancy']>0 for x in row['physical_after']),
                occupied_steps=sum(row['occupancy']>0 for row in run['trace']),
                wall_seconds=run['wall_seconds'],cpu_seconds=run['cpu_seconds'])


def main():
    root=Path('experiments/runs/sensor_restart_short')
    protocol=json.loads((root/'protocol.json').read_text())
    for path,sha in protocol['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('source changed during short study: '+path)
    baseline=Path(protocol['references'][0]);old=Path(protocol['references'][1])
    control=metrics(baseline);previous=metrics(old)
    records={}
    for key in protocol['variants']:
        path=root/(key+'.json');measure=metrics(path)
        paired=compare_probes(path,baseline)
        measure.update(guard_releases=paired['complete_coast_episodes'],
                       guard_releases_without_run=paired['guard_releases_without_ahu_restart'],
                       first_off_excursion_count=paired['first_off_excursion_count'],
                       matched_window_worse_count=paired['matched_window_candidate_worse_count'])
        measure['change_vs_v10_pct']=100*(measure['loss']/control['loss']-1)
        measure['passes_short_gate']=(measure['guard_releases']>0
            and measure['guard_releases_without_run']==0
            and measure['loss']<control['loss']-1e-10
            and measure['occupied_zone_steps_outside']<=control['occupied_zone_steps_outside'])
        records[key]=measure
        write_json(root/(key+'_paired.json'),paired)
    result=dict(status='complete_short_screen',baseline=control,previous_coast=previous,
                candidates=records,next_diagnostic_candidates=[k for k,v in records.items() if v['passes_short_gate']],
                full_sweep_started=False,notebook_exported=False,
                scope='selected diagnostic prefix only; no leaderboard inference')
    write_json(root/'decision.json',result)
    with Path('reports/SENSOR_RESTART_SHORT_REVIEW.md').open('a',encoding='utf-8') as report:
        report.write('\n## Completed short results\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
