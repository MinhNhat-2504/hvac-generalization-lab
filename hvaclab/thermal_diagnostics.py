"""Offline bounded timing and sensor-lag probes; never a submission controller."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import cProfile
import json
import time
from pathlib import Path
import psutil
import numpy as np
from hvaclab.coverage import write_json


def probe(policy_key,building,weather,steps,output):
    import matplotlib
    matplotlib.use('Agg')
    from smart_buildings import evaluate as ev
    from hvaclab.thermal_coast_entry import register
    from hvaclab.registry import resolve_policy
    register()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=Path('smart_buildings/holdout_scenarios/dev')/building/weather
    start=time.perf_counter();cpu=time.process_time()
    env=ev.load_hybrid_environment(config/'config.gin')
    meta=ev.create_building_meta_data(env,config)
    spec=env.action_spec();policy=ev.initialize_policy(resolve_policy(policy_key),meta,spec)
    obs,_=env.reset()
    setup_wall=time.perf_counter()-start;setup_cpu=time.process_time()-cpu
    zones=[(v,i) for ctrl in policy.ahu_controllers for v,i,_ in ctrl['zones']]
    simulator=env.building.simulator
    def truth():
        averages=simulator.building.get_zone_average_temps()
        rooms=simulator.building.room_dict
        values=[]
        for v,_ in zones:
            assigned=[z for z in simulator.hvac.get_zones_for_vav(v) if z in averages]
            area=sum(len(rooms[z]) for z in assigned)
            values.append(sum(averages[z]*len(rooms[z]) for z in assigned)/area)
        return values
    def sensed(x):return [float(policy._denormalize_zone_temp(x[i])) for _,i in zones]
    proc=psutil.Process();rows=[];policy_time=0.;environment_time=0.;diagnostic_time=0.
    profiler=cProfile.Profile();start=time.perf_counter();cpu=time.process_time()
    profiler.enable()
    for step in range(steps):
        t=time.perf_counter();before=sensed(obs);physical_before=truth();diagnostic_time+=time.perf_counter()-t
        t=time.perf_counter();action=ev.validate_action(policy.get_action(obs),spec);policy_time+=time.perf_counter()-t
        t=time.perf_counter();nxt,reward,terminated,truncated,_=env.step(action);environment_time+=time.perf_counter()-t
        t=time.perf_counter()
        rows.append(dict(step=step,elapsed_wall=time.perf_counter()-start,elapsed_cpu=time.process_time()-cpu,
                         observed_before=before,physical_before=physical_before,
                         observed_after=sensed(nxt),physical_after=truth(),reward=float(reward),
                         coasting=list(getattr(policy,'_coasting',[])),
                         ahu_on=[int(action['discrete_action'][c['run_discrete_idx']]) for c in policy.ahu_controllers],
                         occupancy=float(obs[policy.occupancy_idx]),
                         available_gib=psutil.virtual_memory().available/2**30,
                         rss_mib=proc.memory_info().rss/2**20))
        diagnostic_time+=time.perf_counter()-t;obs=nxt
        if terminated or truncated:break
    profiler.disable()
    wall=time.perf_counter()-start;process_cpu=time.process_time()-cpu
    profiler.dump_stats(str(output/(policy_key+'.prof')))
    summary=dict(policy=policy_key,building=building,weather=weather,steps=len(rows),
                 setup_wall=setup_wall,setup_cpu=setup_cpu,wall_seconds=wall,cpu_seconds=process_cpu,
                 policy_seconds=policy_time,environment_seconds=environment_time,
                 diagnostic_seconds=diagnostic_time,comfort_band=list(policy._comfort_band),
                 zone_ids=[v for v,_ in zones],ahu_zone_indices=[[next(j for j,(v,_) in enumerate(zones) if v==zone[0]) for zone in c['zones']] for c in policy.ahu_controllers],
                 source_scope='offline truth diagnostics only; no submission changes',trace=rows)
    write_json(output/(policy_key+'.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='trace'}),flush=True)


def analyze_probe(path):
    """Summarize alignment and coast/restart outcomes without inferring causality."""
    d=json.loads(Path(path).read_text());rows=d['trace'];low,high=d['comfort_band']
    alignment=[];stale=[];episodes=[];active={}
    for r in rows:
        alignment.extend(abs(a-b) for a,b in zip(r['observed_after'],r['physical_before']))
        stale.extend(abs(a-b) for a,b in zip(r['observed_after'],r['physical_after']))
        for handler,coast in enumerate(r['coasting']):
            if coast and handler not in active:active[handler]=r['step']
            if not coast and handler in active:
                begin=active.pop(handler);indices=d['ahu_zone_indices'][handler]
                window=rows[r['step']:min(len(rows),r['step']+7)]
                excursions=[max(low-x,0.,x-high) for w in window for j,x in enumerate(w['physical_after']) if j in indices]
                episodes.append(dict(handler=handler,start_step=begin,restart_step=r['step'],duration_steps=r['step']-begin,
                                     ahu_on_at_guard_release=bool(r['ahu_on'][handler]),
                                     max_excursion_next_35min=max(excursions,default=0.)))
    return dict(policy=d['policy'],steps=d['steps'],wall_seconds=d['wall_seconds'],cpu_seconds=d['cpu_seconds'],
                policy_share=d['policy_seconds']/d['wall_seconds'],environment_share=d['environment_seconds']/d['wall_seconds'],
                max_sensor_error_vs_pre_step=max(alignment,default=0.),
                max_sensor_error_vs_post_step=max(stale,default=0.),
                mean_sensor_error_vs_post_step=float(np.mean(stale)),
                complete_coast_episodes=len(episodes),
                guard_releases_without_ahu_restart=sum(not e['ahu_on_at_guard_release'] for e in episodes),
                episodes_with_excursion_after_restart=sum(e['max_excursion_next_35min']>0 for e in episodes),
                restart_excursions_are_association_not_causation=True,episodes=episodes)


def compare_probes(candidate_path,baseline_path):
    candidate=json.loads(Path(candidate_path).read_text())
    baseline=json.loads(Path(baseline_path).read_text())
    for field in ('building','weather','steps','comfort_band','zone_ids','ahu_zone_indices'):
        if candidate[field]!=baseline[field]:raise ValueError('unmatched probe '+field)
    summary=analyze_probe(candidate_path)
    low,high=candidate['comfort_band']
    for episode in summary['episodes']:
        indices=candidate['ahu_zone_indices'][episode['handler']]
        start=episode['restart_step'];stop=min(candidate['steps'],start+7)
        for key,run in (('candidate',candidate),('baseline',baseline)):
            deviations=[max(low-row['physical_after'][j],0.,row['physical_after'][j]-high)
                        for row in run['trace'][start:stop] for j in indices]
            episode[key+'_window_max_excursion']=max(deviations,default=0.)
        row=candidate['trace'][episode['start_step']]
        episode['first_off_excursion']=max(max(low-row['physical_after'][j],0.,row['physical_after'][j]-high) for j in indices)
    summary['matched_window_candidate_worse_count']=sum(e['candidate_window_max_excursion']>e['baseline_window_max_excursion']+1e-6 for e in summary['episodes'])
    summary['matched_window_baseline_excursion_count']=sum(e['baseline_window_max_excursion']>0 for e in summary['episodes'])
    summary['first_off_excursion_count']=sum(e['first_off_excursion']>0 for e in summary['episodes'])
    summary['baseline_wall_seconds']=baseline['wall_seconds']
    summary['wall_ratio']=candidate['wall_seconds']/baseline['wall_seconds']
    summary['prefix_losses']={key:-sum(r['reward'] for r in run['trace'])/run['steps'] for key,run in (('candidate',candidate),('baseline',baseline))}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=('damper12_mid','coast_lookahead','coast_reserved','coast_empty'),required=True)
    parser.add_argument('--building',default='building_9000');parser.add_argument('--weather',default='atlanta_4')
    parser.add_argument('--steps',type=int,default=576);parser.add_argument('--output',default='experiments/runs/thermal_diagnostics')
    args=parser.parse_args()
    if not 1<=args.steps<=8640:parser.error('steps must be 1..8640')
    probe(args.policy,args.building,args.weather,args.steps,args.output)


if __name__=='__main__':main()
