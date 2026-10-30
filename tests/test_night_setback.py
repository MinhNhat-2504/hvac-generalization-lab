"""The night variant must preserve scored comfort and actually reduce heating."""

import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from test_heat_delivery import building, observation


class NightSetbackTest(unittest.TestCase):
    def make(self, key):
        return resolve_policy(key)(building(), None)

    def test_night_setback_allows_coasting_where_flat_band_heats(self):
        obs = observation([293.5] * 4, hour=23, comfort=False)
        flat, night = self.make("e_sync"), self.make("n_sync_s1")
        self.assertEqual(flat.get_action(obs)["discrete_action"][0], 1)
        np.testing.assert_array_equal(night.get_action(obs)["discrete_action"], [0, 0])
        self.assertEqual(night.comfort_temp_window, (293.0, 297.0))

    def test_scored_window_preserves_all_actions(self):
        flat, night = self.make("e_sync"), self.make("n_sync_s1")
        for temps in ([292, 296, 296, 296], [299] * 4, [295.5] * 4):
            obs = observation(temps)
            left, right = flat.get_action(obs), night.get_action(obs)
            for field in left:
                np.testing.assert_array_equal(left[field], right[field])
        self.assertEqual(night.comfort_temp_window, (294.0, 297.0))

    def test_prestart_ramp_restores_heating_band_toward_deadline(self):
        policy = self.make("n_sync_s1")
        low, high = policy._target_band(observation([295] * 4, hour=5, comfort=False))
        self.assertAlmostEqual(low, 294 - 1 / 3)
        self.assertEqual(high, 297)
        self.assertEqual(policy._target_band(observation([295] * 4, hour=6)), (294, 297))


if __name__ == "__main__":
    unittest.main()
