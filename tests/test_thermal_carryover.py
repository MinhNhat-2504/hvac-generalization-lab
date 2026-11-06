import unittest
import copy
import pandas as pd
from hvaclab.thermal_carryover_analysis import windows,paired_summary,morning_rooms


def row(epoch,temp=296.,run=1):
    return dict(epoch=epoch,zones={'room':dict(temperature=temp,occupancy=1.,low=294.,high=297.,flow=1.)},
        observed_temperature=[temp],comfort=.6*max(0.,temp-297)/5,energy=0.,carbon=0.,
        run=[run],pump=run,hold_left=[100],sat=[-1.],damper=[0.])


class ThermalCarryoverTest(unittest.TestCase):
    def test_naive_clock_stays_local_and_windows_are_contiguous(self):
        epoch=pd.Timestamp('2023-07-20 06:00').timestamp()
        spans=windows(epoch,True,'US/Pacific',6)
        self.assertEqual(pd.Timestamp(spans['prior_afternoon'][0],unit='s').hour,14)
        self.assertEqual(list(spans.values())[-1][1],epoch+7200)
        bounds=list(spans.values())
        for a,b in zip(bounds,bounds[1:]):self.assertEqual(a[1],b[0])

    def test_aware_clock_uses_calendar_local_previous_day(self):
        ts=pd.Timestamp('2023-07-20 06:00',tz='US/Pacific')
        spans=windows(ts.timestamp(),False,'US/Pacific',6)
        start=pd.Timestamp(spans['prior_afternoon'][0],unit='s',tz='UTC').tz_convert('US/Pacific')
        self.assertEqual(str(start),'2023-07-19 14:00:00-07:00')

    def test_half_open_windows_partial_history_and_directional_reward(self):
        base=[row(i*300) for i in range(3)]
        other=[row(i*300,298.,run=0) for i in range(3)]
        a=paired_summary(base,other,0,600)
        self.assertEqual(a['steps'],2);self.assertTrue(a['complete'])
        self.assertAlmostEqual(a['hot_comfort_delta'],.12)
        self.assertEqual(a['cold_comfort_delta'],0)
        self.assertAlmostEqual(a['ahu_machine_hours_delta'],-2/12)
        self.assertFalse(paired_summary(base,other,-300,600)['complete'])
        other[0]['epoch']=1
        with self.assertRaises(ValueError):paired_summary(base,other,0,600)

    def test_identical_trajectories_have_zero_deltas(self):
        base=[row(i*300,298.) for i in range(24)]
        result=paired_summary(base,copy.deepcopy(base),0,7200)
        for field,value in result.items():
            if field.endswith('_delta') or field.endswith('_delta_K'):self.assertEqual(value,0)
        self.assertEqual(morning_rooms(base,base,0)[0]['morning_hot_delta'],0)
