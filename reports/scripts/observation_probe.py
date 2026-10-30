"""Reproduce observation semantics without changing any controller or simulator.

From the repo root, with the project installed:
    python reports/scripts/observation_probe.py
"""

import json
import os
from pathlib import Path


def main():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.simulator.air_handler import AirHandler

    ahu = AirHandler(recirculation=0.7, supply_air_temperature_setpoint=305,
                     fan_static_pressure=200.0, fan_efficiency=0.9)
    sat = dict(reported_sensor_k=ahu.supply_air_temperature_sensor,
               actual_supply_k=ahu.get_supply_air_temp(295.0, 260.0),
               return_k=295.0, outside_k=260.0)
    scenario = Path("smart_buildings/training_scenarios/building_train_7/fairbanks_7")
    env = ev.load_hybrid_environment(scenario / "config.gin")
    meta = ev.create_building_meta_data(env, scenario)
    obs, _ = env.reset()
    idx = meta["zone_temp_indices"][0]
    name = meta["observation_field_names"][idx]
    vav_id = name.removesuffix("_zone_air_temperature_sensor")
    sim = env.building.simulator
    zones = sim.hvac.get_zones_for_vav(vav_id)
    areas = {z: len(sim.building.room_dict[z]) for z in zones}
    norm = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]

    def sensor(vector):
        return float(vector[idx]) * norm["sample_std"] + norm["sample_mean"]

    def actual():
        temps = sim.building.get_zone_average_temps()
        return float(sum(temps[z] * areas[z] for z in zones) / sum(areas.values()))

    spec = env.action_spec()
    action = {"continuous_action": np.ones(spec["continuous_action"].shape, dtype=np.float32),
              "discrete_action": np.ones(spec["discrete_action"].shape, dtype=np.int32)}
    rows = []
    for step in range(3):
        before, observed_before = actual(), sensor(obs)
        obs, _, _, _, _ = env.step(action)
        rows.append(dict(step=step, actual_before_k=before,
                         observed_before_k=observed_before,
                         observed_after_k=sensor(obs), actual_after_k=actual()))
    delayed = (all(abs(r["observed_after_k"] - r["actual_before_k"]) < 1e-4 for r in rows)
               and any(abs(r["actual_after_k"] - r["observed_after_k"]) > 1e-3 for r in rows))
    result = dict(sat_example=sat, scenario=str(scenario), vav=vav_id,
                  zone_observation_matches_preceding_thermal_state=delayed,
                  rows=rows, note="Three-step diagnostic using constant maximum legal actions; "
                                  "not a policy score. The simulator and policies are unmodified.")
    out = Path("experiments/runs/observation_semantics_probe.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
