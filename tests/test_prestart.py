import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class PrestartTest(unittest.TestCase):
    def test_explicit_one_hour_control_matches_v4(self):
        baseline=resolve_policy("e_sync")(building(),None)
        control=resolve_policy("a_start_ctrl")(building(),None)
        for hour in list(range(24))*8:
            obs=observation([294.8,295.6,298.,295.5],hour,comfort=6<=hour<19)
            a,b=baseline.get_action(obs),control.get_action(obs)
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])

    def test_prestart_windows_move_without_changing_comfort_band(self):
        for key, start in (("a_start_ctrl",5),("a_start2",4),("a_start3",3)):
            policy=resolve_policy(key)(building(),None)
            self.assertFalse(policy._in_pi_window(observation([295.5]*4,start-1,False)))
            self.assertTrue(policy._in_pi_window(observation([295.5]*4,start,False)))
            self.assertEqual(policy._comfort_band,(294.,297.))
            self.assertEqual(policy.hold_steps,288)
            self.assertIsNone(policy.pump_pressure)

    def test_extended_prestart_is_still_off_on_weekend(self):
        for key in ("a_start2","a_start3"):
            policy=resolve_policy(key)(building(),None)
            obs=observation([295.5]*4,4,False)
            angle=2*np.pi*(5.25)/7
            obs[2:4]=[np.sin(angle),np.cos(angle)]
            self.assertFalse(policy._in_pi_window(obs))


if __name__ == "__main__":
    unittest.main()
