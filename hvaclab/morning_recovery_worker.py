"""Read-only public/scorer thermal trace around morning recovery."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import json
from pathlib import Path
import sys

from hvaclab.coverage import write_json


def main():
    from smart_buildings import evaluate as ev
    from hvaclab import offhours_hold_worker as base
    initialize, load = ev.initialize_policy, ev.load_hybrid_environment
    state, rows, schema = {}, [], {}

    def traced_initialize(cls, meta, spec):
        policy = initialize(cls, meta, spec)
        if hasattr(policy, 'hold_report'):
            zones = [(v, i, indices) for c in policy.ahu_controllers for v, i, indices in c['zones']]
            schema.update(time_zone=meta['time_zone'], comfort_band=list(policy._comfort_band),
                          morning_start_hour=policy.morning_start_hour,
                          zone_ids=[v for v, _, _ in zones],
                          ahu_ids=[c['ahu_id'] for c in policy.ahu_controllers])
            action = policy.get_action

            def capture(obs):
                command = action(obs)
                state['pending'] = dict(
                    step=len(rows), phase=policy.decision_phase(obs),
                    observed_occupancy=float(obs[policy.occupancy_idx]),
                    observed_temperature=[float(policy._denormalize_zone_temp(obs[i])) for _, i, _ in zones],
                    run=[int(command['discrete_action'][c['run_discrete_idx']]) for c in policy.ahu_controllers],
                    sat=[float(command['continuous_action'][c['temp_idx']]) for c in policy.ahu_controllers],
                    pressure=[float(command['continuous_action'][c['pressure_idx']]) for c in policy.ahu_controllers],
                    damper=[float(command['continuous_action'][a['supply_air_damper_percentage_command']]) for _, _, a in zones],
                    reheat=[float(command['continuous_action'][a['reheat_valve_percentage_command']]) for _, _, a in zones],
                    hold_left=list(policy._hold_left),
                    pump=int(command['discrete_action'][policy.water_discrete_idx]))
                return command
            policy.get_action = capture
        return policy

    def traced_load(*args, **kwargs):
        env = load(*args, **kwargs)
        reward = env.get_reward_info_and_response

        def capture():
            info, response = reward()
            row = state.pop('pending', None)
            if row is not None:
                row.update(epoch=info.start_timestamp.seconds + info.start_timestamp.nanos / 1e9,
                           comfort=-.6 * float(response.normalized_productivity_regret),
                           energy=.2 * float(response.normalized_energy_cost),
                           carbon=.2 * float(response.normalized_carbon_emission),
                           zones={k:dict(temperature=float(v.zone_air_temperature),
                                        occupancy=float(v.average_occupancy),
                                        low=float(v.heating_setpoint_temperature),
                                        high=float(v.cooling_setpoint_temperature),
                                        flow=float(v.air_flow_rate))
                                  for k, v in info.zone_reward_infos.items()})
                rows.append(row)
            return info, response
        env.get_reward_info_and_response = capture
        return env

    ev.initialize_policy, ev.load_hybrid_environment = traced_initialize, traced_load
    try:
        base.main()
    finally:
        ev.initialize_policy, ev.load_hybrid_environment = initialize, load
    job = json.loads(Path(sys.argv[sys.argv.index('--job') + 1]).read_text())
    record = json.loads(Path(job['result_path']).read_text())
    if len(rows) != record['result']['steps']:
        raise ValueError('incomplete thermal trace')
    write_json(Path(job['result_path']).with_suffix('.thermal.json'),
               dict(schema=schema, task=job['task'], fingerprint=job['fingerprint'], rows=rows,
                    semantics='Read-only observer; scorer zone values never enter policy observations.'))
    print('MORNING_THERMAL_TRACE_VERIFIED', len(rows), flush=True)


if __name__ == '__main__':
    main()
