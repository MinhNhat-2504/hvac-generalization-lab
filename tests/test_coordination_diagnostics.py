import unittest

import numpy as np

from hvaclab.heat_diagnostics import coordination_flags
from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class CoordinationDiagnosticsTest(unittest.TestCase):
    def test_observes_held_cooling_reheat_without_water_without_mutation(self):
        policy = resolve_policy("e_sync")(building(), None)
        policy.get_action(observation([299] * 4))
        obs = observation([292, 296, 296, 296])
        action = policy.get_action(obs)
        copied = {k: v.copy() for k, v in action.items()}
        state = (dict(policy._zone_integral), list(policy._hold_left), policy.heat_starts)
        flags = coordination_flags(policy, obs, action)
        self.assertTrue(flags["held_cooling_water_off_cold_reheat"])
        self.assertEqual(state, (policy._zone_integral, policy._hold_left, policy.heat_starts))
        for key in copied:
            np.testing.assert_array_equal(copied[key], action[key])

    def test_idle_heat_override_already_has_water(self):
        policy = resolve_policy("e_sync")(building(), None)
        obs = observation([292, 296, 296, 296])
        flags = coordination_flags(policy, obs, policy.get_action(obs))
        self.assertTrue(flags["running_reheat"])
        self.assertFalse(flags["water_off_running_reheat"])
        self.assertFalse(flags["held_cooling_reheat"])

    def test_in_band_pi_demand_is_not_misreported_as_cold_discomfort(self):
        policy = resolve_policy("e_sync")(building(), None)
        policy.get_action(observation([299] * 4))
        obs = observation([294.5, 296, 296, 296])
        flags = coordination_flags(policy, obs, policy.get_action(obs))
        self.assertTrue(flags["held_cooling_water_off_reheat"])
        self.assertFalse(flags["held_cooling_water_off_cold_reheat"])


if __name__ == "__main__":
    unittest.main()
