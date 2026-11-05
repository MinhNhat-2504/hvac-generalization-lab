import unittest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from hvaclab.morning_recovery_review import summarize,corrected_mornings
from hvaclab.offhours_hold_worker import MorningAccumulator
from hvaclab.policies.morning_precool import MorningPrecoolPolicy
from hvaclab.policies.offhours_hold import OffhoursHoldPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata
from test_heat_delivery import observation


class MorningRecoveryDiagnosticsTest(unittest.TestCase):
    def test_hot_and_cold_occupancy_are_separate_from_nonlinear_comfort(self):
        row=dict(comfort=.02,energy=.003,carbon=.001,run=[1,0],damper=[-1.,1.],reheat=[-1.,0.],
                 zones={'cold':dict(temperature=292.,low=294.,high=297.,occupancy=3.),
                        'hot':dict(temperature=299.,low=294.,high=297.,occupancy=2.),
                        'empty':dict(temperature=285.,low=294.,high=297.,occupancy=0.)})
        r=summarize([row])
        self.assertEqual(r['occupied_cold_degree_person_steps'],6.)
        self.assertEqual(r['occupied_hot_degree_person_steps'],4.)
        self.assertEqual(r['occupant_steps'],5.)
        self.assertEqual(r['comfort_mean'],.02)
        self.assertAlmostEqual(r['ahu_machine_hours'],1/12)
        self.assertAlmostEqual(r['mean_damper'],.5)
        self.assertAlmostEqual(r['mean_reheat'],.25)

    def test_no_trace_before_episode_start_does_not_invent_a_temperature(self):
        self.assertEqual(summarize([]),dict(steps=0))

    def test_naive_local_clock_is_not_shifted_into_previous_date(self):
        epoch=pd.Timestamp('2023-05-30 06:05').timestamp()
        trace=dict(schema=dict(time_zone='US/Pacific'),rows=[dict(epoch=epoch,comfort=.02,
                   zones={'a':dict(occupancy=1.)})])
        days=corrected_mornings(trace,True)
        self.assertEqual(list(days),['2023-05-30'])
        self.assertEqual(days['2023-05-30']['first_local_clock'],'2023-05-30 06:05:00')

    def test_aware_epoch_is_converted_into_building_local_clock(self):
        epoch=pd.Timestamp('2023-05-30 06:05',tz='US/Pacific').timestamp()
        trace=dict(schema=dict(time_zone='US/Pacific'),rows=[dict(epoch=epoch,comfort=.02,
                   zones={'a':dict(occupancy=1.)})])
        days=corrected_mornings(trace,False)
        self.assertEqual(list(days),['2023-05-30'])
        self.assertIn('06:05:00',days['2023-05-30']['first_local_clock'])

    def test_runtime_accumulator_respects_naive_simulator_clock(self):
        tracker=MorningAccumulator('US/Pacific',simulation_clock_naive=True)
        response=SimpleNamespace(total_occupancy=1.,agent_reward_value=-.02,
                                 normalized_productivity_regret=-.02,normalized_energy_cost=.01,normalized_carbon_emission=.01)
        tracker.add(pd.Timestamp('2023-05-30 06:05').timestamp(),response)
        self.assertEqual(list(tracker.days),['2023-05-30'])


class MorningPrecoolPolicyTest(unittest.TestCase):
    def make(self,extra=1.):
        register()
        return MorningPrecoolPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],offhours_hold_steps=6,recovery_extra_hours=extra)

    def obs(self,hour=4,temperatures=None,occupancy=0,weekend=False):
        x=np.r_[observation(temperatures or [296.5]*4,hour,comfort=6<=hour<19),298.,1000.,.7,270.]
        x[5]=occupancy
        if weekend:
            x[2:4]=[np.sin(2*np.pi*5.25/7),np.cos(2*np.pi*5.25/7)]
        return x

    def test_recovery_is_only_in_extra_hour_when_empty_warm_and_not_cold(self):
        for hour,temperatures,occ,weekend,wanted in (
            (4,[296.5]*4,0,False,True),(3,[296.5]*4,0,False,False),
            (4,[295.5]*4,0,False,False),(4,[294.4,297.,297.,297.],0,False,False),
            (4,[296.5]*4,1,False,False),(4,[296.5]*4,0,True,False),
            (4,[float('nan')]*4,0,False,False)):
            p=self.make();x=self.obs(hour,temperatures,occ,weekend);saved=x.copy()
            self.assertEqual(p._early_recovery(x),wanted)
            np.testing.assert_array_equal(x,saved)

    def test_disabled_recovery_matches_short_hold_over_history(self):
        p=self.make(0.);q=OffhoursHoldPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],offhours_hold_steps=6)
        for hour in [23,3,4,5,6,12,19]*10:
            x=self.obs(hour);a,b=p.get_action(x),q.get_action(x)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(p.recovery_steps,0)

    def test_extra_hour_uses_normal_hold_and_continues_integral_into_prestart(self):
        p=self.make();p.get_action(self.obs());self.assertEqual(p.recovery_steps,1)
        self.assertTrue(p._zone_integral);before=dict(p._zone_integral)
        self.assertEqual(p.last_effective_hold_steps,432)
        p.get_action(self.obs(5));self.assertEqual(p.recovery_steps,1)
        self.assertTrue(all(p._zone_integral[k]>=v for k,v in before.items()))
        p.reset();self.assertEqual(p.recovery_steps,0);self.assertFalse(p._zone_integral)

    def test_recovery_duration_is_bounded(self):
        for extra in (-1.,3.,float('nan')):
            with self.assertRaises(ValueError):self.make(extra)


if __name__=='__main__':
    unittest.main()
