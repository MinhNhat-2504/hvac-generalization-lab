import unittest
import numpy as np
from hvaclab.policies.thermal_coasting import ThermalCoastingPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import resolve_policy,_VARIANTS
from test_heat_delivery import building,observation


class ThermalCoastingTest(unittest.TestCase):
    def make(self,mode='all',meta=None,**kwargs):
        register()
        return ThermalCoastingPolicy(meta or building(),None,
                                     **{**_VARIANTS['damper12_mid'][1],'coast_mode':mode,**kwargs})

    def arm(self,p):
        p._hold_left=[100]*len(p.ahu_controllers)
        p._held_action=[(-.5,-1.,1.)]*len(p.ahu_controllers)

    def test_standalone_assembly_matches_all_arms(self):
        from hvaclab.thermal_coast_entry import register,assemble,SPECS
        register()
        for key in SPECS['thermal_coast']['arms']:
            source,_=assemble(key)
            namespace={'__name__':'standalone_coast'}
            exec(compile(source,key,'exec'),namespace)
            packaged=namespace['FinalPolicy'](building(),None)
            local=resolve_policy(key)(building(),None)
            for hour,temperature in [(12,295.5)]*12+[(12,298.)]*8+[(23,295.5)]*12:
                obs=observation([temperature]*4,hour,comfort=hour==12)
                obs[local.occupancy_idx]=1. if hour==12 else 0.
                a,b=local.get_action(obs),packaged.get_action(obs)
                for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_disabled_is_exact_v10(self):
        p=self.make('off')
        q=resolve_policy('damper12_mid')(building(),None)
        for hour,temps in [(12,[292,295,296,299]),(12,[299]*4),(12,[295.5]*4),
                           (23,[294]*4),(5,[292]*4)]*60:
            obs=observation(temps,hour,comfort=hour==12)
            a,b=p.get_action(obs),q.get_action(obs)
            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_coast_respects_minimum_on_freezes_integral_and_closes_water(self):
        p=self.make(minimum_on_steps=3)
        self.arm(p)
        obs=observation([295.5]*4)
        for _ in range(3):self.assertEqual(p.get_action(obs)['discrete_action'][0],1)
        before=dict(p._zone_integral)
        a=p.get_action(obs)
        self.assertEqual(a['discrete_action'].tolist(),[0,0])
        self.assertEqual(p._zone_integral,before)
        self.assertTrue(all(a['continuous_action'][i]==-1 for i in [3,4,5,6,7,8,9,10,11,12]))
        self.assertEqual(p._hold_left,[0])
        self.assertEqual(p.coast_entries,1)
        self.assertEqual(p.get_action(obs)['discrete_action'][0],0)

    def test_boundary_or_fast_trend_restarts_before_observed_temperature_exits_band(self):
        for temps in ([296.4]*4,[296.0]*4):
            p=self.make(minimum_on_steps=1)
            self.arm(p)
            p.get_action(observation([295.5]*4))
            p.get_action(observation([295.5]*4))
            self.assertTrue(p._coasting[0])
            p.get_action(observation(temps))
            self.assertFalse(p._coasting[0])
            self.assertEqual(p.guard_restarts,1)

    def test_empty_only_never_coasts_while_occupied_and_prestart_is_reserved(self):
        for hour,comfort in ((12,True),(5,False)):
            p=self.make('empty',minimum_on_steps=1)
            self.arm(p)
            for _ in range(4):p.get_action(observation([295.5]*4,hour,comfort=comfort))
            self.assertEqual(p.coast_handler_steps,0)
        p=self.make('all',minimum_on_steps=1)
        self.arm(p)
        for _ in range(4):p.get_action(observation([295.5]*4,5,comfort=False))
        self.assertEqual(p.coast_handler_steps,0)

    def test_empty_only_coasts_after_hours_when_empty(self):
        p=self.make('empty',minimum_on_steps=1)
        self.arm(p)
        obs=observation([295.5]*4,23,comfort=False)
        obs[p.occupancy_idx]=0.
        for _ in range(3):a=p.get_action(obs)
        self.assertEqual(a['discrete_action'][0],0)
        self.assertGreater(p.coast_handler_steps,0)

    def test_one_coasting_handler_does_not_stop_shared_pump(self):
        meta=building()
        meta.update(n_discrete_action=3,n_continuous_action=16,temp_indices=[0,3],
                    pressure_indices=[1,4,7],recirculation_indices=[2,5],water_indices=[6],
                    ahu_zone_assignments={'1':['room_0','room_1'],'2':['room_2','room_3']},
                    zone_action_indices={f'VAV_{i}':{'supply_air_damper_percentage_command':8+2*i,
                                                   'reheat_valve_percentage_command':9+2*i} for i in range(4)})
        p=self.make(meta=meta,minimum_on_steps=1)
        self.arm(p)
        for _ in range(3):action=p.get_action(observation([295.5,295.5,292,292]))
        self.assertEqual(action['discrete_action'][0],0)
        self.assertEqual(action['discrete_action'][1],1)
        self.assertEqual(action['discrete_action'][2],1)

    def test_invalid_parameters_rejected(self):
        for params in ({'coast_mode':'bad'},{'coast_margin_k':0.},{'coast_margin_k':2.},
                       {'forecast_steps':0},{'minimum_on_steps':1.5}):
            with self.assertRaises(ValueError):self.make(**params)


if __name__=='__main__':unittest.main()
