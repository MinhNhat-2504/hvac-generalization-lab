"""Regression checks for water coordination and ventilation-loss gating."""

import unittest

import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy
from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class CoordinatedHeatTest(unittest.TestCase):
    def make(self, key, outside=280.0, **kwargs):
        meta = building()
        meta["observation_field_names"].append("ahs_1_outside_air_temperature_sensor")
        obs = np.append(observation([292, 296, 296, 296]), outside)
        if kwargs:
            params = {**resolve_policy(key)._variant_kwargs, **kwargs}
            policy = CoordinatedHeatPolicy(meta, None, **params)
        else:
            policy = resolve_policy(key)(meta, None)
        return policy, obs

    def test_override_starts_water_and_restores_positive_pump_pressure(self):
        policy, obs = self.make("e_sync")
        action = policy.get_action(obs)
        np.testing.assert_array_equal(action["discrete_action"], [1, 1])
        self.assertGreater(action["continuous_action"][4], -1)
        self.assertTrue(np.all(np.isfinite(action["continuous_action"])))
        self.assertEqual(policy.water_sync_steps, 1)

    def test_omitting_water_coordination_reproduces_original(self):
        policy, obs = self.make("e_sync", coordinate_water=False)
        original, _ = self.make("d_heat")
        for _ in range(10):
            left, right = policy.get_action(obs), original.get_action(obs)
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])

    def test_guard_vetoes_high_ventilation_loss(self):
        policy, obs = self.make("e_guard_sync", outside=231.0)
        action = policy.get_action(obs)
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])
        self.assertEqual(policy.heat_guard_vetoes, 1)
        self.assertEqual(policy.water_sync_steps, 0)

    def test_guard_permits_mild_heating(self):
        policy, obs = self.make("e_guard_sync", outside=280.0)
        np.testing.assert_array_equal(policy.get_action(obs)["discrete_action"], [1, 1])

    def test_missing_weather_does_not_force_guarded_run(self):
        policy = resolve_policy("e_guard_sync")(building(), None)
        np.testing.assert_array_equal(
            policy.get_action(observation([292, 296, 296, 296]))["discrete_action"],
            [0, 0],
        )

    def test_pump_economy_cap_is_respected(self):
        policy, obs = self.make("e_sync", pump_pressure=-0.5)
        self.assertEqual(policy.get_action(obs)["continuous_action"][4], -0.5)

    def test_normal_heating_run_is_unchanged(self):
        policy, obs = self.make("e_guard_sync", outside=231.0)
        original, _ = self.make("d_heat", outside=231.0)
        obs[6:10] = 292.0
        left, right = policy.get_action(obs), original.get_action(obs)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(policy.water_sync_steps, 0)

    def test_no_demand_does_not_start_water(self):
        policy, obs = self.make("e_sync")
        obs[6:10] = 295.5
        np.testing.assert_array_equal(policy.get_action(obs)["discrete_action"], [0, 0])


if __name__ == "__main__":
    unittest.main()
