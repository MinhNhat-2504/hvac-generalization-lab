import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


KEYS=("pi_cal_ctrl","pi_cal_slow_i","pi_cal_fast_i","pi_cal_soft_p","pi_cal_firm_p",
      "pi_cal_soft_slow","pi_cal_soft_fast","pi_cal_firm_slow","pi_cal_firm_fast")


class PICalibrationTest(unittest.TestCase):
    def test_control_matches_v4(self):
        baseline=resolve_policy("e_sync")(building(),None)
        control=resolve_policy("pi_cal_ctrl")(building(),None)
        for hour,temp in [(5,[292,296,296,296]),(12,[299]*4),(12,[295.5]*4),(23,[294]*4)]*100:
            obs=observation(temp,hour,comfort=hour==12)
            a,b=baseline.get_action(obs),control.get_action(obs)
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])

    def test_every_gain_retains_full_integral_actuator_authority(self):
        for key in KEYS:
            for sign in (-1.,1.):
                policy=resolve_policy(key)(building(),None)
                policy._zone_integral["VAV_0"]=sign*policy.zone_integral_limit
                action=policy.get_action(observation([295.5]*4))
                self.assertAlmostEqual(policy._zone_u["VAV_0"],sign)
                self.assertEqual(action["continuous_action"][5],1.)
                if sign<0:
                    self.assertEqual(action["continuous_action"][6],1.)

    def test_integrator_stops_winding_up_then_unwinds(self):
        for key in KEYS:
            policy=resolve_policy(key)(building(),None)
            limit=policy.zone_integral_limit
            policy._zone_integral["VAV_0"]=limit
            policy.get_action(observation([296.,295.5,295.5,295.5]))
            self.assertEqual(policy._zone_integral["VAV_0"],limit)
            policy.get_action(observation([295.4,295.5,295.5,295.5]))
            self.assertLess(policy._zone_integral["VAV_0"],limit)


if __name__ == "__main__":
    unittest.main()
