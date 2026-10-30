"""Causal, offline shadow forecast; never supplies temperatures to a controller.

obs[t] holds the preceding thermal state. The change obs[t+1]-obs[t]
belongs to action[t-1], not action[t]. Predictions are evaluated before the
corresponding target is used for learning. No simulator true state is read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def aligned_arrays(observations, continuous, discrete):
    import numpy as np
    obs = np.asarray(observations, dtype=float)
    cont = np.asarray(continuous, dtype=float)
    disc = np.asarray(discrete, dtype=float)
    if len(obs) != len(cont) + 1 or len(cont) != len(disc) or len(cont) < 2:
        raise ValueError("need N actions and N+1 observations, N >= 2")
    return dict(current=obs[1:-1], previous=obs[:-2],
                delta=obs[2:] - obs[1:-1],
                last_continuous=cont[:-1], last_discrete=disc[:-1])


def recursive_predictions(features, targets):
    """Predict first, then learn each target when the next observation arrives."""
    import numpy as np
    x, y = np.asarray(features), np.asarray(targets)
    if x.ndim != 3 or y.shape != x.shape[:2]:
        raise ValueError("need [time, zone, feature] arrays")
    n, zones, dimensions = x.shape
    weights = np.zeros((zones, dimensions))
    covariance = np.repeat(np.eye(dimensions)[None], zones, axis=0)
    predictions = np.empty_like(y)
    for t in range(n):
        vector = x[t]
        predictions[t] = np.einsum("zi,zi->z", weights, vector)
        projected = np.einsum("zij,zj->zi", covariance, vector)
        gain = projected / (1.0 + np.einsum("zi,zi->z", vector, projected))[:, None]
        weights += gain * (y[t] - predictions[t])[:, None]
        covariance -= np.einsum("zi,zj->zij", gain, projected)
    return predictions


def score_forecast(features, targets, trends, warmup=288):
    """Per-zone recursive ridge regression with fixed prior; prequential score."""
    import numpy as np
    y = np.asarray(targets)
    if len(y) <= warmup:
        raise ValueError("need enough post-warmup targets")
    predictions = recursive_predictions(features, targets)
    def metrics(prediction):
        error = prediction[warmup:] - y[warmup:]
        return dict(mae_k=float(np.mean(np.abs(error))),
                    rmse_k=float(np.sqrt(np.mean(error**2))),
                    p95_abs_error_k=float(np.quantile(np.abs(error), .95)))
    return dict(scored_steps=len(y)-warmup, zones=y.shape[1], warmup_steps=warmup,
                persistence=metrics(np.zeros_like(y)),
                constant_trend=metrics(np.asarray(trends)), rls=metrics(predictions))


def evaluate_trace(trace_path, warmup=288):
    import numpy as np
    with np.load(trace_path, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata"]))
        aligned = aligned_arrays(trace["observations"], trace["continuous"], trace["discrete"])
    current, previous = aligned["current"], aligned["previous"]
    norm = metadata["temperature_normalization"]
    scale, offset = norm["sample_std"], norm["sample_mean"]
    features, targets, trends = [], [], []
    feature_names = ["intercept", "zone_minus_centre_div10", "previous_observed_delta_k",
                     "damper_command", "valve_command", "ahu_run", "ahu_sat_command",
                     "ahu_pressure_command", "ahu_recirc_command", "water_run",
                     "water_temperature_command", "water_pressure_command"]
    cont, disc = aligned["last_continuous"], aligned["last_discrete"]
    def column(array, index):
        return np.zeros(len(current)) if index is None else array[:, index]
    for zone in metadata["zones"]:
        idx = zone["observation_index"]
        temperature = current[:, idx] * scale + offset
        trend = (current[:, idx] - previous[:, idx]) * scale
        features.append(np.column_stack([
            np.ones(len(current)), (temperature - metadata["centre_k"]) / 10, trend,
            *[column(cont, zone[k]) for k in ("damper", "valve")],
            column(disc, zone["run"]),
            *[column(cont, zone[k]) for k in ("sat", "pressure", "recirc")],
            column(disc, metadata["water_run"]),
            column(cont, metadata["water_temperature"]),
            column(cont, metadata["water_pressure"])]))
        targets.append(aligned["delta"][:, idx] * scale)
        trends.append(trend)
    result = score_forecast(np.stack(features, axis=1), np.stack(targets, axis=1),
                            np.stack(trends, axis=1), warmup)
    return dict(scenario=metadata["scenario"], policy=metadata["policy"],
                episode_steps=metadata["steps"], mean_reward=metadata["mean_reward"],
                feature_names=feature_names, metrics=result,
                note="Shadow forecast only; short trace is not a full-episode policy score. "
                     "No true simulator state or future action is a feature. "
                     "Good prediction alone does not establish better control.")


def collect(scenario, root, steps, out):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy
    from hvaclab.coverage import digest
    config = Path(root) / scenario
    env = ev.load_hybrid_environment(config / "config.gin")
    meta = ev.create_building_meta_data(env, config)
    spec = env.action_spec()
    policy = ev.initialize_policy(resolve_policy("e_sync"), meta, spec)
    obs, _ = env.reset()
    observations, continuous, discrete, rewards = [np.array(obs, copy=True)], [], [], []
    start = time.monotonic()
    for i in range(steps):
        action = ev.validate_action(policy.get_action(obs), spec)
        continuous.append(np.array(action["continuous_action"], copy=True))
        discrete.append(np.array(action["discrete_action"], copy=True))
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(np.array(obs, copy=True))
        rewards.append(float(reward))
        if (i + 1) % 288 == 0:
            print(f"TRACE {i+1}/{steps} steps; {time.monotonic()-start:.1f}s", flush=True)
        if terminated or truncated:
            break
    zones = []
    for ctrl in policy.ahu_controllers:
        for vav, idx, indices in ctrl["zones"]:
            zones.append(dict(vav=vav, observation_index=idx,
                              damper=indices.get("supply_air_damper_percentage_command"),
                              valve=indices.get("reheat_valve_percentage_command"),
                              run=ctrl["run_discrete_idx"], sat=ctrl["temp_idx"],
                              pressure=ctrl.get("pressure_idx"), recirc=ctrl.get("recirc_idx")))
    metadata = dict(scenario=scenario, scenarios_dir=str(root), policy="e_sync",
                    steps=len(rewards), mean_reward=float(np.mean(rewards)),
                    centre_k=float(sum(policy.comfort_temp_window)/2), zones=zones,
                    temperature_normalization=meta["observation_normalization_metadata"]["zone_air_temperature_sensor"],
                    water_run=policy.water_discrete_idx, water_temperature=policy.water_temp_idx,
                    water_pressure=policy.water_pressure_idx,
                    policy_source_sha256={p.as_posix(): digest(p) for p in Path("hvaclab/policies").glob("*.py")},
                    alignment="predict obs[t+1]-obs[t] using obs through t and action[t-1]")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, observations=np.asarray(observations),
                        continuous=np.asarray(continuous), discrete=np.asarray(discrete),
                        metadata=np.asarray(json.dumps(metadata)))
    return evaluate_trace(out)


def main():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="building_train_7/fairbanks_7")
    parser.add_argument("--scenarios-dir", type=Path, default=Path("smart_buildings/training_scenarios"))
    parser.add_argument("--steps", type=int, default=1440)
    parser.add_argument("--out", type=Path, default=Path("experiments/traces/delay_shadow_fairbanks7.npz"))
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    from hvaclab.coverage import write_json
    result = evaluate_trace(args.evaluate) if args.evaluate else collect(args.scenario, args.scenarios_dir, args.steps, args.out)
    destination = (args.evaluate or args.out).with_suffix(".json")
    write_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
