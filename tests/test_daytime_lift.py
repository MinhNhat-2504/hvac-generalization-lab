import unittest
from unittest.mock import patch
import test_cooling_lift as lift_tests
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
import numpy as np
from hvaclab.policies.daytime_cooling_lift import DaytimeCoolingLiftPolicy
from hvaclab.policies.guarded_cooling_lift import GuardedCoolingLiftPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_cooling_lift import meta
import test_guarded_lift as guarded_tests


class DaytimeLiftTest(unittest.TestCase):
    def make(self,**kwargs):
        register()
        return DaytimeCoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],
            flow_equivalent=True,cooling_lift=True,project_lift=True,**kwargs)

    def test_recovery_and_empty_schedule_match_v10_on_shared_history(self):
        p=self.make(recovery_hours=2)
        q=GuardedCoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],
            flow_equivalent=False,cooling_lift=False,project_lift=True)
        helper=guarded_tests.GuardedLiftTest()
        for h in (p.morning_start_hour-1,p.morning_start_hour,p.morning_start_hour+1,22):
            for t in (296.,297.,298.):
                x=helper.obs(t,hour=h);x[p.comfort_now_idx]=float(h<20)
                if h==22:x[p.comfort_now_idx]=0
                a,b=p.get_action(x),q.get_action(x)
                for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(p.lift_handler_steps,0)
        self.assertTrue(p.cooling_lift and p.flow_equivalent)

    def test_window_boundaries_and_reset(self):
        helper=guarded_tests.GuardedLiftTest()
        for delay in (2.,4.):
            p=self.make(recovery_hours=delay)
            for offset,expected in ((delay-.01,False),(delay,True)):
                x=helper.obs(297.,hour=p.morning_start_hour+offset);x[p.comfort_now_idx]=1
                with patch.object(p,'_get_hour_and_day',return_value=(p.morning_start_hour+offset,0)):
                    self.assertEqual(p._allocation_window(x),expected)
            x[p.comfort_now_idx]=0;self.assertFalse(p._allocation_window(x))
            p.get_action(x);p.reset();self.assertFalse(p._previous_lift_temperatures)

    def test_daytime_still_applies_physical_allocation(self):
        p=self.make(recovery_hours=2)
        helper=guarded_tests.GuardedLiftTest()
        with patch.object(FlowEquivalentPolicy,'get_action',side_effect=lambda obs:lift_tests.CoolingLiftTest().command()):
            p.get_action(helper.obs(296.9));p.get_action(helper.obs(297.))
        self.assertGreater(p.lift_handler_steps,0)
        self.assertTrue(p.last_allocations)

    def test_invalid_delay(self):
        for delay in (-1,5,float('nan')):
            with self.assertRaises(ValueError):self.make(recovery_hours=delay)
