import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
from hvaclab.policies.offhours_hold import OffhoursHoldPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata
from test_heat_delivery import observation


class FlowEquivalentTest(unittest.TestCase):
    def make(self,enabled=True,supported=True):
        register();m=metadata()
        if supported:m['action_normalization_metadata']['1_supply_air_static_pressure_setpoint']=dict(min_native_value=0.,max_native_value=2000.,min_normalized_value=-1.,max_normalized_value=1.)
        return FlowEquivalentPolicy(m,None,**_VARIANTS['damper12_mid'][1],flow_equivalent=enabled)

    def obs(self):return np.r_[observation([299.]*4),298.,1000.,.7,270.]

    def test_flow_preserved_across_random_pressures_and_damper_floor(self):
        p=self.make();rng=np.random.default_rng(42)
        for _ in range(500):
            command=dict(continuous_action=rng.uniform(-1,1,13).astype(np.float32),discrete_action=np.array([1,1]))
            command['continuous_action'][1]=rng.uniform(-1,1)
            with patch.object(OffhoursHoldPolicy,'get_action',return_value=command):
                action=p.get_action(self.obs())
            old=p.last_parent_action['continuous_action'];new=action['continuous_action']
            a=np.maximum(.1,(old[[5,7,9,11]].astype(float)+1)/2)*np.sqrt(p.native_pressure(0,float(old[1])))
            b=np.maximum(.1,(new[[5,7,9,11]].astype(float)+1)/2)*np.sqrt(p.native_pressure(0,float(new[1])))
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8)
            self.assertLessEqual(float(new[1]),float(old[1])+1e-6)
            np.testing.assert_array_equal(new[[0,2,3,4,6,8,10,12]],old[[0,2,3,4,6,8,10,12]])
        self.assertGreater(p.rebalanced_handler_steps,0)

    def test_disabled_and_missing_metadata_match_v10_over_history(self):
        for p in (self.make(False),self.make(supported=False)):
            q=OffhoursHoldPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1])
            for _ in range(30):
                x=self.obs();saved=x.copy();a,b=p.get_action(x),q.get_action(x)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
                np.testing.assert_array_equal(x,saved)

    def test_closed_damper_floor_is_compensated_and_fan_power_decreases(self):
        p=self.make();command=dict(continuous_action=np.full(13,-1.,dtype=np.float32),discrete_action=np.array([1,0]))
        command['continuous_action'][1]=0.
        with patch.object(OffhoursHoldPolicy,'get_action',return_value=command):a=p.get_action(self.obs())
        self.assertAlmostEqual(p.native_pressure(0,float(a['continuous_action'][1])),200.,places=3)
        self.assertGreater(a['continuous_action'][5],-1.)
        self.assertEqual(p.rebalanced_handler_steps,1)
        p.reset();self.assertEqual(p.rebalanced_handler_steps,0);self.assertIsNone(p.last_parent_action)

    def test_off_handler_commands_are_not_rewritten(self):
        p=self.make();command=dict(continuous_action=np.zeros(13,dtype=np.float32),discrete_action=np.array([0,0]))
        with patch.object(OffhoursHoldPolicy,'get_action',return_value=command):a=p.get_action(self.obs())
        np.testing.assert_array_equal(a['continuous_action'],0.)
        self.assertEqual(p.rebalanced_handler_steps,0)


if __name__=='__main__':unittest.main()
