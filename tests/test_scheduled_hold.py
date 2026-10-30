import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class ScheduledHoldTest(unittest.TestCase):
    def test_control_matches_v4_across_day_and_night(self):
        candidate = resolve_policy("s_hold_ctrl")(building(), None)
        baseline = resolve_policy("e_sync")(building(), None)
        for hour, temps in [(12, [299]*4), (23, [295.5]*4),
                            (5, [295.5]*4), (12, [292]*4)]*100:
            obs = observation(temps, hour, comfort=hour == 12)
            a, b = candidate.get_action(obs), baseline.get_action(obs)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_no_hold_still_honors_real_night_cooling_call(self):
        policy = resolve_policy("s_night0")(building(), None)
        action = policy.get_action(observation([299]*4, 23, comfort=False))
        self.assertEqual(action["discrete_action"][0], 1)
        self.assertEqual(policy._hold_left[0], 0)
        action = policy.get_action(observation([295.5]*4, 23, comfort=False))
        self.assertEqual(action["discrete_action"][0], 0)

    def test_night_hold_expires_and_daytime_hold_remains(self):
        policy = resolve_policy("s_night12")(building(), None)
        hot = observation([299]*4, 23, comfort=False)
        calm = observation([295.5]*4, 23, comfort=False)
        policy.get_action(hot)
        self.assertEqual(policy._hold_left[0], 12)
        for _ in range(12):
            self.assertEqual(policy.get_action(calm)["discrete_action"][0], 1)
        self.assertEqual(policy.get_action(calm)["discrete_action"][0], 0)
        policy.get_action(observation([299]*4))
        self.assertEqual(policy._hold_left[0], 288)
        policy.get_action(observation([295.5]*4))
        self.assertEqual(policy._hold_left[0], 287)


if __name__ == "__main__":
    unittest.main()
