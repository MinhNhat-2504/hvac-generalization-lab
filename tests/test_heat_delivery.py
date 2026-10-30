"""Action-level checks for the airflow-only heating experiment.

Use synthetic building metadata so these checks exercise the real policy chain
without running or changing a simulation that is already in progress.
"""

import unittest

import numpy as np

from hvaclab.registry import resolve_policy


def building(zone_count=4):
    names = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
             "comfort_mode_now", "num_occupants"]
    names += [f"VAV_{i}_zone_air_temperature_sensor" for i in range(zone_count)]
    return {
        "comfort_temp_window": (294.0, 297.0),
        "eco_temp_window": (289.0, 298.0),
        "morning_start_hour": 6, "evening_start_hour": 19,
        "n_discrete_action": 2, "n_continuous_action": 5 + 2 * zone_count,
        "temp_indices": [0], "pressure_indices": [1, 4],
        "recirculation_indices": [2], "water_indices": [3],
        "zone_action_indices": {
            f"VAV_{i}": {"supply_air_damper_percentage_command": 5 + 2 * i,
                          "reheat_valve_percentage_command": 6 + 2 * i}
            for i in range(zone_count)
        },
        "observation_field_names": names,
        "ahu_zone_assignments": {"1": [f"room_{i}" for i in range(zone_count)]},
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 0.0, "sample_std": 1.0}
        },
    }


def observation(temperatures, hour=12, comfort=True):
    angle = 2 * np.pi * (hour + 0.25) / 24
    return np.array([np.sin(angle), np.cos(angle), 0.0, 1.0,
                     float(comfort), 1.0, *temperatures])


class HeatDeliveryTest(unittest.TestCase):
    def policy(self, key):
        return resolve_policy(key)(building(), None)

    def assert_actions_equal(self, left, right):
        for key in ("continuous_action", "discrete_action"):
            np.testing.assert_array_equal(left[key], right[key])

    def test_control_reproduces_h288_over_changing_demands(self):
        control, reference = self.policy("d_ctrl"), self.policy("h_288")
        for hour, temps in [(12, [292, 296, 296, 296]), (12, [299] * 4),
                            (12, [295.5] * 4), (12, [292] * 4),
                            (23, [295.5] * 4)] * 4:
            obs = observation(temps, hour, comfort=hour == 12)
            self.assert_actions_equal(control.get_action(obs), reference.get_action(obs))

    def test_cold_outlier_gets_minimum_airflow_without_changing_vav_or_water(self):
        obs = observation([292, 296, 296, 296])
        control = self.policy("d_ctrl").get_action(obs)
        policy = self.policy("d_heat")
        action = policy.get_action(obs)
        self.assertEqual(control["discrete_action"][0], 0)
        self.assertEqual(action["discrete_action"][0], 1)
        np.testing.assert_array_equal(action["continuous_action"][:3], [1, -1, 1])
        np.testing.assert_array_equal(action["continuous_action"][3:],
                                      control["continuous_action"][3:])
        self.assertEqual(action["discrete_action"][1], control["discrete_action"][1])
        self.assertEqual(policy.heat_starts, 1)

    def test_fraction_threshold_is_inclusive(self):
        obs = observation([292, 296, 296, 296])
        self.assertEqual(self.policy("d_heat25").get_action(obs)["discrete_action"][0], 1)
        self.assertEqual(self.policy("d_heat50").get_action(obs)["discrete_action"][0], 0)

    def test_existing_heating_run_is_untouched(self):
        obs = observation([292] * 4)
        self.assert_actions_equal(self.policy("d_heat").get_action(obs),
                                  self.policy("d_ctrl").get_action(obs))

    def test_no_heat_demand_does_not_start_idle_handler(self):
        obs = observation([295.5] * 4)
        policy = self.policy("d_heat")
        self.assertEqual(policy.get_action(obs)["discrete_action"][0], 0)
        self.assertEqual(policy.heat_starts, 0)

    def test_outside_pi_window_does_not_force_heat(self):
        obs = observation([292, 296, 296, 296], hour=23, comfort=False)
        self.assert_actions_equal(self.policy("d_heat").get_action(obs),
                                  self.policy("d_ctrl").get_action(obs))

    def test_existing_cooling_hold_is_untouched(self):
        heat, control = self.policy("d_heat"), self.policy("d_ctrl")
        for temps in ([299] * 4, [292, 296, 296, 296]):
            obs = observation(temps)
            self.assert_actions_equal(heat.get_action(obs), control.get_action(obs))
        self.assertEqual(heat.holds_applied, 1)
        self.assertEqual(heat.heat_starts, 0)


if __name__ == "__main__":
    unittest.main()
