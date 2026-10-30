"""Measure whether airflow overrides also have hot water available.

This observes the existing policy without changing its commands. A short probe
is a mechanism check, not a full-episode leaderboard estimate.

    python -m hvaclab.heat_diagnostics --steps 1440 --out experiments/traces/heat_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def coordination_flags(policy, obs, action):
    """Observe current PI reheat commands without advancing controller state.

    A held cooling run is identified by a running, cooling AHU whose inherited
    aggregate error is zero. This diagnostic targets the heat variants with
    force_ahu_on=False; it is not a general classifier for arbitrary policies.
    """
    flags = dict(pi_active=bool(policy._pi_active), running_reheat=False,
                 water_off_running_reheat=False, water_off_cold_reheat=False,
                 held_cooling_reheat=False, held_cooling_water_off_reheat=False,
                 held_cooling_water_off_cold_reheat=False)
    if not policy._pi_active:
        return flags
    cont, disc = action["continuous_action"], action["discrete_action"]
    water_off = disc[policy.water_discrete_idx] == 0
    low, _ = policy._comfort_band
    for ctrl in policy.ahu_controllers:
        if not disc[ctrl["run_discrete_idx"]]:
            continue
        requesting = []
        for vav, idx, indices in ctrl["zones"]:
            valve_idx = indices.get("reheat_valve_percentage_command")
            if (policy._zone_u.get(vav, 0) < 0 and valve_idx is not None
                    and float(cont[valve_idx]) > -1.0):
                requesting.append(policy._denormalize_zone_temp(obs[idx]) < low)
        if not requesting:
            continue
        flags["running_reheat"] = True
        cold = any(requesting)
        flags["water_off_running_reheat"] |= bool(water_off)
        flags["water_off_cold_reheat"] |= bool(water_off and cold)
        held = (not policy.force_ahu_on and float(cont[ctrl["temp_idx"]]) < 0
                and policy._ahu_error(obs, ctrl, *policy.comfort_temp_window) == 0)
        flags["held_cooling_reheat"] |= held
        flags["held_cooling_water_off_reheat"] |= bool(held and water_off)
        flags["held_cooling_water_off_cold_reheat"] |= bool(held and water_off and cold)
    return flags


def probe(policy_key: str, scenario: str, steps: int, scenarios_dir=None) -> dict:
    from hvaclab.runner import FULL_EPISODE_STEPS, SCENARIOS_DIR
    import matplotlib
    matplotlib.use("Agg")
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy

    root = SCENARIOS_DIR if scenarios_dir is None else Path(scenarios_dir)
    config_dir = root / scenario
    env = ev.load_hybrid_environment(config_dir / "config.gin")
    meta = ev.create_building_meta_data(env, config_dir)
    policy = ev.initialize_policy(resolve_policy(policy_key), meta, env.action_spec())
    if not hasattr(policy, "heat_starts"):
        raise ValueError("choose a HeatDeliveryPolicy variant, such as d_heat")
    obs, _ = env.reset()
    counts = dict(steps=0, override_steps=0, handler_overrides=0,
                  overrides_with_water_off=0, overrides_with_min_water_temperature=0,
                  overrides_with_min_water_pressure=0)
    coordination = {}
    total_reward = 0.0
    for _ in range(steps):
        before = policy.heat_starts
        action = ev.validate_action(policy.get_action(obs), env.action_spec())
        for key, active in coordination_flags(policy, obs, action).items():
            coordination[key] = coordination.get(key, 0) + int(active)
        overrides = policy.heat_starts - before
        if overrides:
            counts["override_steps"] += 1
            counts["handler_overrides"] += overrides
            cont, disc = action["continuous_action"], action["discrete_action"]
            if disc[policy.water_discrete_idx] == 0:
                counts["overrides_with_water_off"] += 1
            if policy.water_temp_idx is not None and cont[policy.water_temp_idx] == -1:
                counts["overrides_with_min_water_temperature"] += 1
            if policy.water_pressure_idx is not None and cont[policy.water_pressure_idx] == -1:
                counts["overrides_with_min_water_pressure"] += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        counts["steps"] += 1
        total_reward += float(reward)
        if terminated or truncated:
            break
    return {
        "policy": policy_key, "scenario": scenario, "scenarios_dir": root.as_posix(),
        "requested_steps": steps, **counts,
        "coordination_step_counts": coordination,
        "full_episode": counts["steps"] == FULL_EPISODE_STEPS,
        "mean_reward": total_reward / counts["steps"],
        "note": "Water counters count steps containing overrides, not handlers. "
                "Minimum temperature is 310 K and may still provide some heat; "
                "water off or zero pump pressure prevents reheat flow. "
                "Coordination counters observe current PI commands; cold means "
                "below the comfort lower bound in the pre-action observation. "
                "A short run must not be compared with full-episode scores.",
    }


def main() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from hvaclab.runner import FULL_EPISODE_STEPS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="d_heat")
    parser.add_argument("--scenario", default="building_train_7/fairbanks_7")
    parser.add_argument("--steps", type=int, default=FULL_EPISODE_STEPS)
    parser.add_argument("--scenarios-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.steps <= FULL_EPISODE_STEPS:
        parser.error(f"--steps must be between 1 and {FULL_EPISODE_STEPS}")
    result = probe(args.policy, args.scenario, args.steps, args.scenarios_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
