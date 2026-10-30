"""Full-episode clock-aligned comfort-direction and actuator diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timezone
import json
import os
from pathlib import Path
import time


def split_comfort_loss(info,weighted_loss,reward_function):
    dt=(info.end_timestamp.seconds-info.start_timestamp.seconds)+(info.end_timestamp.nanos-info.start_timestamp.nanos)*1e-9
    cold=hot=0.
    for z in info.zone_reward_infos.values():
        if z.average_occupancy<=0:
            continue
        full=reward_function._get_zone_productivity_reward(
            z.heating_setpoint_temperature,z.cooling_setpoint_temperature,
            .5*(z.heating_setpoint_temperature+z.cooling_setpoint_temperature),dt,z.average_occupancy)
        actual=reward_function._get_zone_productivity_reward(
            z.heating_setpoint_temperature,z.cooling_setpoint_temperature,
            z.zone_air_temperature,dt,z.average_occupancy)
        if z.zone_air_temperature < z.heating_setpoint_temperature:
            cold+=max(0.,full-actual)
        elif z.zone_air_temperature > z.cooling_setpoint_temperature:
            hot+=max(0.,full-actual)
    total=cold+hot
    if total<=0:
        return 0.,0.
    return weighted_loss*cold/total,weighted_loss*hot/total


def profile(scenario, root, policy_key, out):
    import matplotlib
    matplotlib.use("Agg")
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy
    from hvaclab.heat_diagnostics import coordination_flags
    from hvaclab.runner import ScenarioResult
    from hvaclab.coverage import write_json, source_snapshot, digest
    plan=json.loads(Path("experiments/plans/e_sync_coverage.json").read_text())
    snapshot=source_snapshot(plan)
    snapshot["hvaclab/comfort_profile.py"]=digest(__file__)
    snapshot["hvaclab/heat_diagnostics.py"]=digest("hvaclab/heat_diagnostics.py")
    building,weather=scenario.split("/")
    config=Path(root)/scenario
    started=time.monotonic()
    env=ev.load_hybrid_environment(config/"config.gin")
    meta=ev.create_building_meta_data(env,config)
    spec=env.action_spec()
    policy=ev.initialize_policy(resolve_policy(policy_key),meta,spec)
    hours=[dict(steps=0,occupied_steps=0,reward=0.,comfort=0.,energy=0.,carbon=0.) for _ in range(24)]
    result=ScenarioResult(building,weather,policy_key,0,0.,0.)
    occupancy_total=0.
    cold_loss=hot_loss=0.
    mechanism_counts={}
    mechanism_comfort={}
    time_samples=[]
    obs,_=env.reset()
    for step in range(8640):
        action=ev.validate_action(policy.get_action(obs),spec)
        flags=coordination_flags(policy,obs,action)
        low,high=policy._comfort_band
        hot_low=saturated=False
        if policy._pi_active:
            for ctrl in policy.ahu_controllers:
                pressure=ctrl.get("pressure_idx")
                if (not action["discrete_action"][ctrl["run_discrete_idx"]]
                        or action["continuous_action"][ctrl["temp_idx"]] >= 0
                        or pressure is None or action["continuous_action"][pressure] > -.8):
                    continue
                for vav,idx,_ in ctrl["zones"]:
                    if policy._denormalize_zone_temp(obs[idx]) > high and policy._zone_u.get(vav,0)>0:
                        hot_low=True
                        saturated |= policy._zone_u.get(vav,0)>=.8
        flags["hot_pi_low_cooling_pressure"]=hot_low
        flags["saturated_hot_pi_low_cooling_pressure"]=saturated
        obs,reward,terminated,truncated,_=env.step(action)
        info,rr=env.get_reward_info_and_response()
        # Match the actual policy clock, which uses simulation wall-clock
        # features. Converting naive-simulation proto timestamps to US/Pacific
        # would shift the bins by seven hours in this scenario.
        hour,_=policy._get_hour_and_day(obs)
        row=hours[hour]
        comfort=-.6*float(rr.normalized_productivity_regret)
        cold,hot=split_comfort_loss(info,comfort,env.reward_function)
        cold_loss+=cold
        hot_loss+=hot
        for name,active in flags.items():
            mechanism_counts[name]=mechanism_counts.get(name,0)+int(active)
            mechanism_comfort[name]=mechanism_comfort.get(name,0.)+(comfort if active else 0.)
        if step in (0,71,72,1440):
            time_samples.append(dict(step=step,simulation_timestamp=str(env.current_simulation_timestamp),
                                     reward_start_utc=str(info.start_timestamp.ToDatetime(tzinfo=timezone.utc)),
                                     policy_hour=hour,comfort_mode=bool(obs[policy.comfort_now_idx]>.5),
                                     metadata_time_zone=str(meta["time_zone"])))
        row["steps"]+=1
        row["reward"]+=float(reward)
        row["comfort"]+=-.6*float(rr.normalized_productivity_regret)
        row["energy"]+=.2*float(rr.normalized_energy_cost)
        row["carbon"]+=.2*float(rr.normalized_carbon_emission)
        result.steps+=1
        result.total_reward+=float(reward)
        result.prod_regret+=float(rr.normalized_productivity_regret)
        result.energy_cost+=float(rr.normalized_energy_cost)
        result.carbon+=float(rr.normalized_carbon_emission)
        occupancy=float(rr.total_occupancy)
        occupancy_total+=occupancy
        if occupancy>0:
            result.occupied_steps+=1
            result.reward_occupied+=float(reward)
            row["occupied_steps"]+=1
        else:
            result.reward_unoccupied+=float(reward)
        if (step+1)%1440==0:
            print(f"PROFILE {step+1}/8640; {time.monotonic()-started:.1f}s",flush=True)
        if terminated or truncated:
            break
    result.mean_reward=result.total_reward/result.steps
    result.mean_occupancy=occupancy_total/result.steps
    result.seconds=time.monotonic()-started
    opening=meta["morning_start_hour"]
    closing=meta["evening_start_hour"]
    total_comfort=sum(r["comfort"] for r in hours)
    early=sum(hours[h%24]["comfort"] for h in range(opening,opening+2))
    fraction=early/total_comfort if total_comfort else 0.
    payload=dict(result=asdict(result),hourly=hours,morning_start_hour=opening,
                 cold_comfort_loss=cold_loss,hot_comfort_loss=hot_loss,
                 mechanism_step_counts=mechanism_counts,mechanism_comfort_loss=mechanism_comfort,
                 time_samples=time_samples,
                 evening_start_hour=closing,first_two_hours_comfort_share=fraction,
                 first_two_hours_uniform_share=2/(closing-opening),source_sha256=snapshot,
                 note="Full-episode diagnostic aligned to observed policy wall-clock features; unchanged control actions. "
                      "Hour bins aggregate workdays and unoccupied days; comfort is occupancy-weighted.")
    if result.steps!=8640:
        raise RuntimeError("short diagnostic episode is not a valid reference")
    write_json(out,payload)
    print(json.dumps(dict(scenario=scenario,score=result.score,
                          first_two_hours_comfort_share=fraction,seconds=result.seconds)),flush=True)


def main():
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario",default="building_train_7/phoenix_3")
    parser.add_argument("--scenarios-dir",type=Path,default=Path("smart_buildings/training_scenarios"))
    parser.add_argument("--policy",default="a_start_ctrl")
    parser.add_argument("--out",type=Path,default=Path("experiments/runs/prestart_profile_policyclock.json"))
    args=parser.parse_args()
    profile(args.scenario,args.scenarios_dir,args.policy,args.out)


if __name__ == "__main__":
    main()
