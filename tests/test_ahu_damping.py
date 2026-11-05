import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.policies.ahu_damped import AHUDampedPolicy
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata
from test_heat_delivery import observation


def meta():
    m=metadata();m['action_normalization_metadata']['1_supply_air_static_pressure_setpoint']=dict(min_native_value=0.,max_native_value=2000.,min_normalized_value=-1.,max_normalized_value=1.)
    return m


class AHUDampingTest(unittest.TestCase):
    def make(self,**kwargs):
        register();return AHUDampedPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],**kwargs)

    def obs(self,temp=299.):return np.r_[observation([temp]*4),298.,1000.,.7,270.]

    def test_disabled_is_exact_flow_baseline_through_history(self):
        p=self.make(ahu_damping=False);q=FlowEquivalentPolicy(meta(),None,**_VARIANTS['damper12_mid'][1])
        for temp in (299.,295.5,292.,296.)*60:
            x=self.obs(temp);saved=x.copy();a,b=p.get_action(x),q.get_action(x)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(x,saved)

    def test_filter_bound_and_sign_zero_urgent_bypasses(self):
        p=self.make();c=p.ahu_controllers[0];x=self.obs()
        def call(raw):
            with patch.object(FlowEquivalentPolicy,'_ahu_error',return_value=raw):return p._ahu_error(x,c,294.,297.)
        self.assertEqual(call(.1),.1)
        self.assertAlmostEqual(call(.9),.75)
        self.assertAlmostEqual(p.ahu_max_applied_correction,.15)
        self.assertEqual(call(1.2),1.2)
        self.assertEqual(call(-.3),-.3)
        self.assertEqual(call(0.),0.)
        self.assertEqual(call(.4),.4)
        self.assertGreater(p.ahu_filtered_steps,0)
        self.assertGreater(p.ahu_urgent_bypasses,0)

    def test_zone_pi_actions_integrals_and_observation_are_unchanged_on_shared_history(self):
        p=self.make(flow_equivalent=False);q=self.make(flow_equivalent=False,ahu_damping=False)
        for temp in (297.2,297.4,297.8,297.1,296.7,298.5)*20:
            x=self.obs(temp);a,b=p.get_action(x),q.get_action(x)
            np.testing.assert_array_equal(a['continuous_action'][5:],b['continuous_action'][5:])
            self.assertEqual(p._zone_integral,q._zone_integral)
        self.assertGreater(p.ahu_filtered_steps,0)

    def test_identity_settings_and_reset_erase_history(self):
        for options in (dict(ahu_alpha=1.),dict(ahu_max_correction_k=0.)):
            p=self.make(**options);q=self.make(ahu_damping=False)
            for temp in (297.2,297.8,297.3)*10:
                a,b=p.get_action(self.obs(temp)),q.get_action(self.obs(temp))
                for key in a:np.testing.assert_array_equal(a[key],b[key])
        p=self.make();p.get_action(self.obs(297.2));p.get_action(self.obs(297.8));p.reset()
        self.assertFalse(p._ahu_filtered);self.assertEqual(p.ahu_filtered_steps,0)

    def test_invalid_settings_are_rejected(self):
        for options in (dict(ahu_alpha=0),dict(ahu_alpha=2),dict(ahu_alpha=float('nan')),
                        dict(ahu_max_correction_k=-1),dict(ahu_urgent_k=0)):
            with self.assertRaises(ValueError):self.make(**options)

    def test_short_gate_rejects_comfort_regression_despite_lower_total(self):
        from hvaclab.ahu_damping_review import gate
        means={'phase_ctrl':dict(loss=.02,comfort=.003),'flow_reset':dict(loss=.019,comfort=.0031),
               'flow_smooth':dict(loss=.018,comfort=.0032)}
        cases={'flow_smooth':{'b/phoenix_3':dict(diagnostics=dict(ahu_filtered_steps=10),
            morning_vs_v10=dict(no_worse_each_morning=True),candidate=dict(loss=.018),v10=dict(loss=.02))}}
        self.assertFalse(gate(means,cases)['passes_short_gate'])
        means['flow_smooth']['comfort']=.0029
        self.assertTrue(gate(means,cases)['passes_short_gate'])
        cases['flow_smooth']['b/phoenix_3']['morning_vs_v10']['no_worse_each_morning']=False
        self.assertFalse(gate(means,cases)['passes_short_gate'])


if __name__=='__main__':unittest.main()
