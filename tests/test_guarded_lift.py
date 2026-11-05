import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.policies.guarded_cooling_lift import GuardedCoolingLiftPolicy
from hvaclab.policies.cooling_lift import CoolingLiftPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_cooling_lift import meta
from test_heat_delivery import observation


class GuardedLiftTest(unittest.TestCase):
    def make(self,**kwargs):
        register();return GuardedCoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=True,cooling_lift=True,**kwargs)
    def obs(self,t,hour=12):return np.r_[observation([t]*4,hour,comfort=hour==12),298.,1000.,.7,310.,4.,.2,[2.]*4]

    def test_disabled_projection_and_guard_match_prior_policy_on_shared_history(self):
        p=self.make();q=CoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=True,cooling_lift=True)
        for t in (296.,297.,298.,295.,294.7)*40:
            x=self.obs(t);saved=x.copy();a,b=p.get_action(x),q.get_action(x)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(x,saved)

    def test_projection_is_signed_causal_bounded_and_reset_erases_history(self):
        p=self.make(project_lift=True);p.get_action(self.obs(296.))
        self.assertFalse(p._lift_projection)
        p.get_action(self.obs(296.2))
        self.assertAlmostEqual(p._lift_projection['VAV_0'],.4,places=4)
        p.get_action(self.obs(297.));self.assertEqual(p._lift_projection['VAV_0'],.5)
        p.get_action(self.obs(296.));self.assertEqual(p._lift_projection['VAV_0'],-.5)
        p.reset();self.assertFalse(p._previous_lift_temperatures);self.assertEqual(p.max_projection_k,0)

    def test_guard_restores_same_step_v10_and_removes_physical_allocation_record(self):
        p=self.make(project_lift=True,comfort_guard=True)
        parent=dict(continuous_action=np.zeros(13,dtype=np.float32),discrete_action=np.array([1,0]))
        def fake_action(obs):
            p.last_parent_action={k:v.copy() for k,v in parent.items()}
            p.last_allocations={'1':dict()};p.lift_handler_steps+=1
            return dict(continuous_action=np.full(13,-.7,dtype=np.float32),discrete_action=np.array([1,0]))
        p._previous_lift_temperatures={f'VAV_{i}':296.5 for i in range(4)}
        with patch.object(CoolingLiftPolicy,'get_action',side_effect=fake_action):a=p.get_action(self.obs(297.))
        indices=[0,1,2,5,6,7,8,9,10,11,12]
        np.testing.assert_array_equal(a['continuous_action'][indices],parent['continuous_action'][indices])
        self.assertFalse(p.last_allocations);self.assertEqual(p.lift_handler_steps,0);self.assertEqual(p.comfort_guard_steps,1)

    def test_guard_does_not_override_unoccupied_offhours(self):
        p=self.make(project_lift=True,comfort_guard=True)
        p.get_action(self.obs(296.,23));p.get_action(self.obs(298.,23))
        # Synthetic fixture has one public occupant unless explicitly cleared.
        x=self.obs(299.,23);x[5]=0.;before=p.comfort_guard_steps;p.get_action(x)
        self.assertEqual(p.comfort_guard_steps,before)

    def test_cold_crossing_triggers_same_step_v10_fallback(self):
        p=self.make(project_lift=True,comfort_guard=True)
        p._previous_lift_temperatures={f'VAV_{i}':295. for i in range(4)}
        def action(obs):
            p.last_parent_action=dict(continuous_action=np.zeros(13,dtype=np.float32),discrete_action=np.array([1,0]))
            return dict(continuous_action=np.full(13,-.7,dtype=np.float32),discrete_action=np.array([1,0]))
        with patch.object(CoolingLiftPolicy,'get_action',side_effect=action):a=p.get_action(self.obs(294.6))
        self.assertEqual(p.comfort_guard_steps,1);self.assertEqual(a['continuous_action'][0],0.)

    def test_score_saving_does_not_override_comfort_or_morning_gate(self):
        from hvaclab.guarded_lift_review import candidate_gate
        means={'phase_ctrl':dict(loss=.02,comfort=.003),'flow_lift':dict(loss=.0184,comfort=.0032),
               'guard_lift':dict(loss=.0189,comfort=.0031)}
        cases={'guard_lift':{'b/phoenix_5':dict(diagnostics=dict(lift_handler_steps=10,projected_zone_steps=30),
            morning_vs_v10=dict(no_worse_each_morning=True),candidate=dict(loss=.018),v10=dict(loss=.02))}}
        self.assertFalse(candidate_gate('guard_lift',means,cases)['passes_short_gate'])
        means['guard_lift']['comfort']=.0029
        self.assertTrue(candidate_gate('guard_lift',means,cases)['passes_short_gate'])
        cases['guard_lift']['b/phoenix_5']['morning_vs_v10']['no_worse_each_morning']=False
        self.assertFalse(candidate_gate('guard_lift',means,cases)['passes_short_gate'])

    def test_invalid_limits_are_rejected(self):
        for options in (dict(projection_limit_k=.6),dict(projection_steps=3),dict(guard_margin_k=-1)):
            with self.assertRaises(ValueError):self.make(**options)


if __name__=='__main__':unittest.main()
