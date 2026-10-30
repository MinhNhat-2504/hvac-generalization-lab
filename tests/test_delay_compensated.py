import unittest

import numpy as np

from hvaclab.policies.delay_compensated import DelayCompensatedPolicy
from hvaclab.registry import resolve_policy
from hvaclab.delay_shadow import recursive_predictions
from test_heat_delivery import building, observation


class DelayCompensatedTest(unittest.TestCase):
    def test_online_predictions_match_causal_shadow_model(self):
        policy = DelayCompensatedPolicy(building(), None, prediction_blend=1.,
                                        prediction_limit_k=100., prediction_warmup=0)
        features, predictions, temperatures = [], [], []
        for t in range(91):
            temp = np.array([295 + np.sin(t*.07+i) for i in range(4)])
            temperatures.append(temp)
            policy.get_action(observation(temp))
            if policy._previous_features is not None:
                features.append(policy._previous_features.copy())
                predictions.append(policy._correction_k.copy())
        targets = np.asarray(temperatures)[2:] - np.asarray(temperatures)[1:-1]
        expected = recursive_predictions(np.asarray(features)[:-1], targets)
        np.testing.assert_allclose(np.asarray(predictions)[:-1], expected, atol=1e-12)

    def test_zero_blend_reproduces_reference_actions_after_learning(self):
        reference = resolve_policy("e_sync")(building(), None)
        candidate = resolve_policy("r_zero")(building(), None)
        for t in range(600):
            temperatures = [295 + 3*np.sin(t*.13+i) for i in range(4)]
            hour = 12 if t % 200 < 150 else 23
            obs = observation(temperatures, hour, comfort=hour == 12)
            saved = obs.copy()
            left, right = reference.get_action(obs), candidate.get_action(obs)
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
            np.testing.assert_array_equal(obs, saved)
        self.assertTrue(np.any(candidate._weights != 0))

    def test_warmup_and_prediction_bounds(self):
        policy = resolve_policy("r_half")(building(), None)
        for t in range(289):
            policy.get_action(observation([294 + np.sin(t*.2)]*4))
            if t < 288:
                np.testing.assert_array_equal(policy._correction_k, np.zeros(4))
        self.assertGreater(np.max(np.abs(policy._correction_k)), 0)
        self.assertLessEqual(np.max(np.abs(policy._correction_k)), .5)

    def test_prediction_is_applied_only_to_pi_zone_commands(self):
        kwargs = dict(setback_k=0., setup_k=0., heating_pressure=-1.,
                      cooling_pressure_gain=1., reheat_full_k=.5, hold_steps=288)
        predicted = DelayCompensatedPolicy(building(), None, prediction_warmup=0, **kwargs)
        reference = resolve_policy("e_sync")(building(), None)
        obs = observation([295.5]*4, 23, comfort=False)
        for t in range(12):
            obs[6:] = 295.5 + .1*t
            action = predicted.get_action(obs)
            baseline = reference.get_action(obs)
            for key in baseline:
                np.testing.assert_array_equal(action[key], baseline[key])

    def test_estimator_reset_discards_prior_episode(self):
        policy = resolve_policy("r_half")(building(), None)
        for t in range(10):
            policy.get_action(observation([294+t*.1]*4))
        policy.reset()
        self.assertIsNone(policy._previous_features)
        self.assertIsNone(policy._previous_action)
        self.assertEqual(policy._prediction_steps, 0)
        np.testing.assert_array_equal(policy._weights, np.zeros((4,12)))


if __name__ == "__main__":
    unittest.main()
