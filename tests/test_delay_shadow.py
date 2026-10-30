import unittest

import numpy as np

from hvaclab.delay_shadow import aligned_arrays, recursive_predictions


class DelayShadowTest(unittest.TestCase):
    def test_delayed_impulse_is_paired_with_previous_action(self):
        # Action 0 causes a thermal jump, first visible in observation 2.
        obs = np.array([[0.], [0.], [2.], [2.], [5.]])
        commands = np.array([[2.], [0.], [3.], [99.]])
        aligned = aligned_arrays(obs, commands, np.ones_like(commands))
        np.testing.assert_array_equal(aligned["delta"], [[2.], [0.], [3.]])
        np.testing.assert_array_equal(aligned["last_continuous"], [[2.], [0.], [3.]])

    def test_future_targets_cannot_change_earlier_predictions(self):
        x = np.ones((8, 2, 3))
        targets = np.arange(16).reshape(8, 2).astype(float)
        original = recursive_predictions(x, targets)
        altered = targets.copy()
        altered[4:] += 1000
        predictions = recursive_predictions(x, altered)
        np.testing.assert_array_equal(predictions[:5], original[:5])
        self.assertFalse(np.array_equal(predictions[5:], original[5:]))


if __name__ == "__main__":
    unittest.main()
