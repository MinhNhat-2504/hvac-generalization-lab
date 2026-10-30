"""Full-episode reward-by-hour diagnostic, without changing control actions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timezone
import json
import os
from pathlib import Path
import time


def profile(scenario, root, policy_key, out):
    import matplotlib
    matplotlib.use("Agg")
    import pytz
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy
    from hvaclab.runner import ScenarioResult
    from hvaclab.coverage import write_json, source_snapshot, digest
    plan=json.loads(Path("experiments/plans/e_sync_coverage.json").read_text())
    snapshot=source_snapshot(plan)
    snapshot["hvaclab/morning_profile.py"]=digest(__file__)
    building,weather=scenario.split("/")
    config=Path(root)/scenario
    started=time.monotonic()
    env=ev.load_hybrid_environment(config/"config.gin")
    meta=ev.create_building_meta_data(env,config)
    spec=env.action_spec()
    policy=ev.initialize_policy(resolve_policy(policy_key),meta,spec)
    zone=meta["time_zone"]
    if isinstance(zone,str):
        zone=pytz.timezone(zone)
    hours=[dict(steps=0,occupied_steps=0,reward=0.,comfort=0.,energy=0.,carbon=0.) for _ in range(24)]
    result=ScenarioResult(building,weather,policy_key,0,0.,0.)
    occupancy_total=0.
    obs,_=env.reset()
    for step in range(8640):
        action=ev.validate_action(policy.get_action(obs),spec)
        obs,reward,terminated,truncated,_=env.step(action)
        info,rr=env.get_reward_info_and_response()
        # Reward timestamps, converted to the scenario schedule's timezone;
        # the diagnostic does not read hidden thermal state.
        when=info.start_timestamp.ToDatetime(tzinfo=timezone.utc).astimezone(zone)
        row=hours[when.hour]
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
                 evening_start_hour=closing,first_two_hours_comfort_share=fraction,
                 first_two_hours_uniform_share=2/(closing-opening),source_sha256=snapshot,
                 note="Full-episode diagnostic using actual reward timestamps; unchanged control actions. "
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
    parser.add_argument("--out",type=Path,default=Path("experiments/runs/prestart_profile.json"))
    args=parser.parse_args()
    profile(args.scenario,args.scenarios_dir,args.policy,args.out)


if __name__ == "__main__":
    main()
