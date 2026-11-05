import unittest
import numpy as np
from hvaclab.policies.comfort_band_pi import ComfortBandPIPolicy
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata
from test_heat_delivery import observation


class ComfortBandPITest(unittest.TestCase):
    def make(self,**kwargs):
        register();return ComfortBandPIPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=False,**kwargs)

    def obs(self,t,hour=12,comfort=True):return np.r_[observation([t]*4,hour,comfort),298.,1000.,.7,270.]

    def test_disabled_and_collapsed_band_reproduce_v10_actions(self):
        for options in (dict(band_pi=False),dict(comfort_reserve_k=1.5)):
            p=self.make(**options);q=FlowEquivalentPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=False)
            for t in (292.,294.6,295.5,296.4,299.)*40:
                x=self.obs(t);saved=x.copy();a,b=p.get_action(x),q.get_action(x)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
                np.testing.assert_array_equal(x,saved)

    def test_inside_band_keeps_integral_and_delivers_existing_heating_load(self):
        p=self.make();p._zone_integral={f'VAV_{i}':-60. for i in range(4)}
        x=self.obs(294.7);p.get_action(x)
        for i in range(4):
            self.assertEqual(p._zone_integral[f'VAV_{i}'],-60.)
            self.assertAlmostEqual(p._zone_u[f'VAV_{i}'],-.3)
        self.assertEqual(p.inside_band_with_memory_steps,4)

    def test_error_has_correct_sign_and_reduction_is_bounded_by_inner_halfwidth(self):
        for temperature,wanted in ((293.5,-.105),(297.5,.105),(295.5,0.)):
            p=self.make();p.get_action(self.obs(temperature))
            for u in p._zone_u.values():self.assertAlmostEqual(u,wanted)
            self.assertLessEqual(p.max_error_reduction_k,1.+1e-10)

    def test_outside_pi_window_is_unchanged_and_clears_integral_as_parent(self):
        p=self.make();q=self.make(band_pi=False)
        for t in (292.,295.7,299.)*10:
            a,b=p.get_action(self.obs(t,23,False)),q.get_action(self.obs(t,23,False))
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(p.band_zone_steps,0);self.assertFalse(p._zone_integral)

    def test_reset_and_bad_reserves(self):
        p=self.make();p.get_action(self.obs(294.));self.assertGreater(p.band_zone_steps,0)
        p.reset();self.assertEqual(p.band_zone_steps,0);self.assertFalse(p._zone_integral)
        for reserve in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.make(comfort_reserve_k=reserve)

    def test_nontrivial_observation_normalization_preserves_physical_band_error(self):
        register();m=metadata()
        m['observation_normalization_metadata']['zone_air_temperature_sensor']=dict(sample_mean=295.,sample_std=2.)
        p=ComfortBandPIPolicy(m,None,**_VARIANTS['damper12_mid'][1],flow_equivalent=False)
        x=self.obs(297.5);x[6:10]=(297.5-295.)/2;saved=x.copy();p.get_action(x)
        for u in p._zone_u.values():self.assertAlmostEqual(u,.105)
        np.testing.assert_array_equal(x,saved)

    def test_short_gate_requires_material_loss_gain_and_morning_comfort(self):
        from hvaclab.comfort_band_review import candidate_gate
        means={'phase_ctrl':dict(loss=.02,comfort=.003),'flow_reset':dict(loss=.0195,comfort=.0031),
               'band_flow':dict(loss=.0198,comfort=.0029)}
        cases={'band_flow':{'a/w':dict(diagnostics=dict(band_zone_steps=10),
            morning_vs_v10=dict(no_worse_each_morning=True),candidate=dict(loss=.018),v10=dict(loss=.02))}}
        self.assertFalse(candidate_gate('band_flow',means,cases)['passes_short_gate'])
        means['band_flow']['loss']=.0185
        self.assertTrue(candidate_gate('band_flow',means,cases)['passes_short_gate'])
        cases['band_flow']['a/w']['morning_vs_v10']['no_worse_each_morning']=False
        self.assertFalse(candidate_gate('band_flow',means,cases)['passes_short_gate'])


if __name__=='__main__':unittest.main()
