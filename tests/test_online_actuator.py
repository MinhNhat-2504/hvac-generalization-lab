import unittest
import numpy as np
from hvaclab.policies.online_actuator import OnlineActuatorEstimator,OnlineActuatorPolicy
from hvaclab.policies.actuator_thermal_model import FEATURE_NAMES
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS,resolve_policy
from test_heat_delivery import building,observation


def metadata():
    m=building();m['observation_field_names']+=['ahs_1_'+s for s in ('supply_air_temperature_sensor','supply_air_static_pressure_sensor','recirculation_percentage_command','outside_air_temperature_sensor')]
    m['action_normalization_metadata']={'supply_water_temperature_setpoint':dict(min_native_value=300.,max_native_value=360.,min_normalized_value=-1.,max_normalized_value=1.)}
    return m


def obs(t):return np.r_[observation([295.5+np.sin(t*.08+i)*.2 for i in range(4)]),298.,1000.,.7,270.]


class OnlineActuatorTest(unittest.TestCase):
    def make(self,blend=0.):
        register();return OnlineActuatorPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],online_blend=blend,online_warmup=0)

    def test_shadow_is_exact_v10_after_online_learning(self):
        p=self.make();q=resolve_policy('damper12_mid')(metadata(),None)
        for t in range(180):
            x=obs(t);saved=x.copy();a,b=p.get_action(x),q.get_action(x)
            for field in a:np.testing.assert_array_equal(a[field],b[field])
            np.testing.assert_array_equal(x,saved)
        self.assertTrue(p._supported)
        self.assertGreater(p.estimator.counts.sum(),0)
        self.assertEqual(p.correction_zone_steps,0)

    def test_estimator_updates_only_after_next_sensor_arrives(self):
        p=OnlineActuatorEstimator(1,minimum_samples=1);x=np.zeros((1,len(FEATURE_NAMES)));x[0,0]=1.
        p.predict(np.array([295.]),np.array([0.]),x,['on_steady'],1.)
        self.assertEqual(p.counts.sum(),0)
        np.testing.assert_array_equal(p.weights,0.)
        p.observe(np.array([295.2]));self.assertEqual(p.counts.sum(),1)
        self.assertGreater(p.weights[0,6,0],0.)
        self.assertIsNone(p.pending)

    def test_pending_predictions_are_pre_update_error_scores(self):
        p=OnlineActuatorEstimator(1,minimum_samples=1);x=np.zeros((1,len(FEATURE_NAMES)));x[0,0]=1.
        p.counts[0,6]=1
        p.predict(np.array([295.]),np.array([.1]),x,['on_steady'],1.)
        p.observe(np.array([295.2]))
        self.assertAlmostEqual(p.error[6,2],.2)
        self.assertGreater(p.weights[0,6,0],0.)

    def test_reset_erases_learned_state_and_corrections_are_bounded(self):
        p=self.make(.25)
        for t in range(180):p.get_action(obs(t))
        self.assertTrue(np.all(np.abs(p._correction)<=.5))
        self.assertGreater(p.estimator.counts.sum(),0)
        p.reset();self.assertEqual(p.estimator.counts.sum(),0)
        self.assertIsNone(p._last_action);self.assertIsNone(p.estimator.pending)
        np.testing.assert_array_equal(p.estimator.weights,0.)

    def test_selected_feedback_is_applied_and_clamped(self):
        p=self.make(1.)
        p.get_action(obs(0));p.get_action(obs(1))
        p.estimator.counts[:]=100;p.estimator.error_steps[:]=100
        p.estimator.error[:]=[1.,0.,2.]
        temperatures=np.array([p._denormalize_zone_temp(obs(2)[j]) for _,j,_,_ in p._online_zones])
        p._previous_temps=temperatures-100.
        x=obs(2);saved=x.copy();p.get_action(x)
        np.testing.assert_allclose(p._correction,.5)
        self.assertGreater(p.correction_zone_steps,0)
        np.testing.assert_array_equal(x,saved)

    def test_missing_public_fields_falls_back_to_v10(self):
        register();m=building();p=OnlineActuatorPolicy(m,None,**_VARIANTS['damper12_mid'][1],online_blend=.25)
        q=resolve_policy('damper12_mid')(m,None)
        for _ in range(4):
            x=observation([295.5]*4);a,b=p.get_action(x),q.get_action(x)
            for field in a:np.testing.assert_array_equal(a[field],b[field])
        self.assertFalse(p._supported)


if __name__=='__main__':unittest.main()
