import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.zone_reserve_policy import ZoneReservePolicy
from hvaclab.policies.guarded_cooling_lift import GuardedCoolingLiftPolicy
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
import test_cooling_lift as lift
import test_guarded_lift as guarded


class ZoneReserveTest(unittest.TestCase):
    def make(self,fraction=.2):
        register()
        p=ZoneReservePolicy(lift.meta(),None,**_VARIANTS['damper12_mid'][1],
            flow_equivalent=False,cooling_lift=True,project_lift=True,reserve_fraction=fraction)
        p._pi_active=True;p._zone_u={f'VAV_{i}':.5 for i in range(4)}
        return p

    def test_zero_reserve_matches_unrestricted_projection(self):
        p=self.make(0);q=GuardedCoolingLiftPolicy(lift.meta(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=False,cooling_lift=True,project_lift=True)
        for t in (296.,297.,298.,295.)*20:
            obs=guarded.GuardedLiftTest().obs(t);a,b=p.get_action(obs),q.get_action(obs)
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_hot_zone_reserve_and_non_hot_target_are_separate_and_bounded(self):
        p=self.make();obs=lift.CoolingLiftTest().obs();obs[6:10]=[297.,295.,298.,296.6]
        with patch.object(FlowEquivalentPolicy,'get_action',side_effect=lambda obs:lift.CoolingLiftTest().command()):p.get_action(obs)
        allocation=p.last_allocations['1'];mult=allocation['target_cooling_multiplier']
        self.assertAlmostEqual(mult[0],1.2);self.assertEqual(mult[1],1)
        self.assertAlmostEqual(mult[2],1.2);self.assertGreater(mult[3],1)
        for j,factor in enumerate(mult):
            before=allocation['old_flows'][j]*(allocation['planning_temperature'][j]-allocation['old_supply'])
            after=allocation['realized_flows'][j]*(allocation['planning_temperature'][j]-allocation['new_supply'])
            self.assertAlmostEqual(after/before,factor,places=5)
        self.assertLessEqual(max(mult),1.2);self.assertEqual(p.reserve_zone_steps,3)

    def test_capacity_veto_keeps_original_allocation(self):
        p=self.make();obs=lift.CoolingLiftTest().obs();obs[6:10]=305.
        command=lift.CoolingLiftTest().command()
        # Existing allocation fits this public capacity; reserve would exceed it.
        obs[-6]=.6;obs[-5]=.2
        with patch.object(FlowEquivalentPolicy,'get_action',side_effect=lambda obs:command.copy()):p.get_action(obs)
        self.assertGreaterEqual(p.reserve_vetoes,1)
        self.assertNotIn('target_cooling_multiplier',p.last_allocations['1'])

    def test_no_positive_pi_or_inactive_means_no_reserve(self):
        for active,u in ((True,0.),(False,.5)):
            p=self.make();p._pi_active=active;p._zone_u={f'VAV_{i}':u for i in range(4)}
            with patch.object(FlowEquivalentPolicy,'get_action',side_effect=lambda obs:lift.CoolingLiftTest().command()):p.get_action(lift.CoolingLiftTest().obs())
            self.assertEqual(p.reserve_zone_steps,0)

    def test_reset_and_invalid_budget(self):
        p=self.make();p.reserve_zone_steps=4;p.reset();self.assertEqual(p.reserve_zone_steps,0)
        for fraction in (-.1,.3,float('nan')):
            with self.assertRaises(ValueError):self.make(fraction)
