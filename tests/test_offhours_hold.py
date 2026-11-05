from types import SimpleNamespace
import unittest
import numpy as np
from hvaclab.policies.offhours_hold import OffhoursHoldPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS,resolve_policy
from hvaclab.offhours_hold_worker import MorningAccumulator
from test_online_actuator import metadata
from test_heat_delivery import observation


class OffhoursHoldTest(unittest.TestCase):
    def make(self,cap):
        register();return OffhoursHoldPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],offhours_hold_steps=cap)

    def obs(self,temperature=299.,hour=23,comfort=False,occupancy=0):
        x=np.r_[observation([temperature]*4,hour,comfort=comfort),298.,1000.,.7,270.]
        x[5]=occupancy;return x

    def test_control_matches_v10_and_does_not_mutate_observation(self):
        p=self.make(None);q=resolve_policy('damper12_mid')(metadata(),None)
        for hour,comfort,temp in [(23,False,299.),(23,False,295.5),(5,False,293.),(12,True,299.)]*20:
            x=self.obs(temp,hour,comfort,1 if comfort else 0);saved=x.copy();a,b=p.get_action(x),q.get_action(x)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(x,saved)

    def test_refresh_is_short_only_when_empty_and_pi_window_inactive(self):
        for hour,comfort,occupancy,expected in ((23,False,0,6),(23,False,1,432),(5,False,0,432),(12,True,0,432),(12,True,1,432)):
            p=self.make(6);p.get_action(self.obs(299.,hour,comfort,occupancy))
            self.assertEqual(p._hold_left[0],expected)
            self.assertEqual(p.hold_steps,432)

    def test_carried_memory_is_not_trimmed_and_parent_request_is_not_vetoed(self):
        p=self.make(6);p._hold_left=[100];p._held_action=[(-.5,-1.,1.)]
        a=p.get_action(self.obs(295.5))
        self.assertEqual(p._hold_left[0],99);self.assertEqual(a['discrete_action'][0],1)
        q=self.make(1)
        for _ in range(12):
            a=q.get_action(self.obs(299.));self.assertEqual(a['discrete_action'][0],1);self.assertEqual(q._hold_left[0],1)

    def test_next_protected_request_refreshes_normal_duration(self):
        p=self.make(6);p.get_action(self.obs(299.));self.assertEqual(p._hold_left[0],6)
        p.get_action(self.obs(299.,5,False,0));self.assertEqual(p._hold_left[0],432)

    def test_morning_window_begins_at_first_occupied_reward_interval(self):
        tracker=MorningAccumulator('UTC')
        def response(occupancy):return SimpleNamespace(total_occupancy=occupancy,agent_reward_value=-.01,
            normalized_productivity_regret=-.001,normalized_energy_cost=.01,normalized_carbon_emission=.01)
        for t,o in ((0,0),(300,1),(600,0),(7200,1),(7500,1),(86400+300,1)):tracker.add(t,response(o))
        self.assertEqual(tracker.days['1970-01-01']['steps'],3)
        self.assertEqual(tracker.days['1970-01-01']['occupied_steps'],2)
        self.assertEqual(tracker.days['1970-01-02']['steps'],1)

    def test_each_morning_guard_cannot_hide_a_bad_day_in_the_mean(self):
        from hvaclab.offhours_hold_review import morning_comparison
        def day(regret,worst=.02):return dict(first_occupied_epoch=0.,steps=24,occupied_steps=20,prod_regret=regret,worst_occupied_comfort=worst)
        baseline=dict(morning_comfort={'a':day(-.24),'b':day(-.24)})
        candidate=dict(morning_comfort={'a':day(-.20),'b':day(-.28)})
        result=morning_comparison(baseline,candidate)
        self.assertFalse(result['no_worse_each_morning'])
        self.assertEqual([r['day'] for r in result['failed_mornings']],['b'])

    def test_morning_comparison_requires_identical_occupancy_exposure(self):
        from hvaclab.offhours_hold_review import morning_comparison
        row=dict(first_occupied_epoch=0.,steps=24,occupied_steps=20,prod_regret=-.24,worst_occupied_comfort=.02)
        baseline=dict(morning_comfort={'a':dict(row)})
        candidate=dict(morning_comfort={'a':{**row,'first_occupied_epoch':300.}})
        with self.assertRaises(ValueError):morning_comparison(baseline,candidate)

    def test_invalid_caps_are_rejected(self):
        for cap in (0,-1,True,1.5):
            with self.assertRaises(ValueError):self.make(cap)


if __name__=='__main__':unittest.main()
