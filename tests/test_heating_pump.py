import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class HeatingPumpTest(unittest.TestCase):
    def test_uncapped_control_matches_reference(self):
        baseline = resolve_policy("e_sync")(building(), None)
        control = resolve_policy("v_pump_ctrl")(building(), None)
        for hour, temp in [(12,[299]*4), (12,[294.8,295.6,295.6,295.6]),
                           (23,[292]*4), (12,[296]*4)]*100:
            obs=observation(temp,hour,comfort=hour==12)
            a,b=baseline.get_action(obs),control.get_action(obs)
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])

    def test_cooling_hold_blocks_pressure_economy_even_with_reheat(self):
        for key in ("v_pump_warm", "v_pump_heat"):
            baseline=resolve_policy("e_sync")(building(),None)
            candidate=resolve_policy(key)(building(),None)
            for obs in (observation([299]*4),observation([294.8,295.6,295.6,295.6])):
                a,b=baseline.get_action(obs),candidate.get_action(obs)
                for field in a:
                    np.testing.assert_array_equal(a[field],b[field])
            self.assertEqual(candidate.pump_caps,0)

    def test_override_arm_limits_only_pressure_on_safe_heating_override(self):
        baseline=resolve_policy("e_sync")(building(),None)
        candidate=resolve_policy("v_pump_heat")(building(),None)
        obs=observation([294.8,295.6,295.6,295.6])
        a,b=baseline.get_action(obs),candidate.get_action(obs)
        self.assertGreater(candidate.heat_starts,0)
        self.assertEqual(b["continuous_action"][4],0)
        expected=a["continuous_action"].copy()
        expected[4]=0
        np.testing.assert_array_equal(expected,b["continuous_action"])
        np.testing.assert_array_equal(a["discrete_action"],b["discrete_action"])

    def test_cold_or_high_demand_keeps_reference_pressure(self):
        baseline=resolve_policy("e_sync")(building(),None)
        candidate=resolve_policy("v_pump_heat")(building(),None)
        for obs in (observation([292,296,296,296]), observation([292]*4)):
            a,b=baseline.get_action(obs),candidate.get_action(obs)
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])


if __name__ == "__main__":
    unittest.main()
