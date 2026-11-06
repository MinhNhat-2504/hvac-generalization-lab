import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from hvaclab.policies.cutoff_cooling_lift import CutoffCoolingLiftPolicy
from hvaclab.policies.daytime_cooling_lift import DaytimeCoolingLiftPolicy
from hvaclab.cutoff_lift_worker import observe_action
from test_cooling_lift import meta
import test_guarded_lift as helpers


class CutoffLiftTest(unittest.TestCase):
    def make(self,cutoff=14):
        register()
        return CutoffCoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],
            flow_equivalent=True,cooling_lift=True,project_lift=True,recovery_hours=2.,cutoff_hour=cutoff)

    def test_cutoff_boundaries_morning_and_comfort_gate(self):
        for cutoff in (14,16):
            p=self.make(cutoff);obs=helpers.GuardedLiftTest().obs(297.)
            for hour,expected in ((7,False),(8,True),(cutoff-1,True),(cutoff,False),(23,False)):
                with patch.object(p,'_get_hour_and_day',return_value=(hour,0)):
                    self.assertEqual(p._allocation_window(obs),expected)
            obs[p.comfort_now_idx]=0
            with patch.object(p,'_get_hour_and_day',return_value=(10,0)):self.assertFalse(p._allocation_window(obs))

    def test_before_cutoff_matches_two_hour_control_on_shared_history(self):
        p=self.make();q=DaytimeCoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],
            flow_equivalent=True,cooling_lift=True,project_lift=True,recovery_hours=2.)
        for t in (296.,297.,298.,295.)*10:
            obs=helpers.GuardedLiftTest().obs(t);a,b=p.get_action(obs),q.get_action(obs)
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_after_cutoff_returns_same_step_parent_without_resetting_pi(self):
        p=self.make();p._zone_integral={'VAV_0':10.}
        obs=helpers.GuardedLiftTest().obs(297.,hour=16);obs[p.comfort_now_idx]=1
        command=p.get_action(obs)
        for key in command:np.testing.assert_array_equal(command[key],p.last_parent_action[key])
        self.assertFalse(p.last_allocations)
        self.assertTrue(p._zone_integral)
        self.assertTrue(p.flow_equivalent and p.cooling_lift)

    def test_observer_is_read_only_and_copies_state(self):
        p=self.make();q=self.make();rows=[];observed=observe_action(p,p.get_action,rows)
        for hour in (12,16,22,12):
            obs=helpers.GuardedLiftTest().obs(297.,hour=hour);obs[p.comfort_now_idx]=float(hour!=22)
            saved=obs.copy();a,b=observed(obs),q.get_action(obs)
            np.testing.assert_array_equal(obs,saved)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(p._zone_integral,q._zone_integral)
            self.assertEqual(p._hold_left,q._hold_left)
        frozen=dict(rows[0]['integral_after']);p._zone_integral.clear()
        self.assertEqual(rows[0]['integral_after'],frozen)
        self.assertEqual(rows[1]['integral_before'],rows[0]['integral_after'])

    def test_bad_cutoff_rejected(self):
        for cutoff in (13,15,float('nan')):
            with self.assertRaises(ValueError):self.make(cutoff)
