import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class GuardedPumpTest(unittest.TestCase):
    def policies(self):
        return [resolve_policy(k)(building(), None) for k in
                ("e_sync", "u_pump_ctrl", "u_pump_cap", "u_pump_guard")]

    def test_control_matches_v4_over_changing_demands(self):
        baseline, control, _, _ = self.policies()
        for hour, temp in [(12,[292]*4), (12,[299]*4), (23,[295.5]*4),
                           (12,[294.8,295.6,295.6,295.6])]*100:
            obs = observation(temp, hour, comfort=hour==12)
            left, right = baseline.get_action(obs), control.get_action(obs)
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])

    def test_cold_zone_retains_reference_pressure_after_water_sync(self):
        baseline, _, cap, guard = self.policies()
        obs = observation([292,296,296,296])
        ref = baseline.get_action(obs)
        limited, protected = cap.get_action(obs), guard.get_action(obs)
        self.assertEqual(ref["discrete_action"][1], 1)
        self.assertEqual(limited["continuous_action"][4], 0)
        for key in ref:
            np.testing.assert_array_equal(ref[key], protected[key])
        self.assertEqual(guard.pump_guard_restores, 1)

    def test_safe_open_valve_changes_only_pump_pressure(self):
        baseline, _, _, guard = self.policies()
        obs = observation([294.8,295.6,295.6,295.6])
        saved = obs.copy()
        ref, action = baseline.get_action(obs), guard.get_action(obs)
        self.assertEqual(ref["discrete_action"][1],1)
        self.assertEqual(action["continuous_action"][4],0)
        expected = ref["continuous_action"].copy()
        expected[4] = 0
        np.testing.assert_array_equal(expected,action["continuous_action"])
        np.testing.assert_array_equal(ref["discrete_action"],action["discrete_action"])
        np.testing.assert_array_equal(obs,saved)

    def test_large_valve_request_restores_pressure_even_if_zone_is_in_band(self):
        baseline, _, _, guard = self.policies()
        for policy in (baseline, guard):
            policy._zone_integral["VAV_0"] = -190
        obs = observation([295.4]*4)
        ref, action = baseline.get_action(obs), guard.get_action(obs)
        for key in ref:
            np.testing.assert_array_equal(ref[key],action[key])
        self.assertEqual(guard.pump_guard_restores,1)


if __name__ == "__main__":
    unittest.main()
