import copy
import unittest
import numpy as np
from hvaclab.plant_runtime_telemetry import stage_taps,aggregate,metered_watts
from hvaclab.policies.phase_online_actuator import PhaseOnlineActuatorPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata
from test_heat_delivery import observation
from smart_buildings.smart_control.proto.smart_control_reward_pb2 import RewardInfo


class PlantRuntimeTest(unittest.TestCase):
    def make(self):
        register();return PhaseOnlineActuatorPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],online_blend=0.)

    def observation(self,temperatures):return np.r_[observation(temperatures),298.,1000.,.7,270.]

    def test_real_hold_override_is_distinct_from_armed_memory(self):
        p=self.make();p._hold_left=[3];p._held_action=[(-.5,-1.,1.)]
        with stage_taps(lambda:p) as frame:
            action=p.get_action(self.observation([295.5]*4))
            self.assertEqual(frame['parent']['run'],[0])
            self.assertEqual(frame['hold'],[True]);self.assertEqual(frame['heat_override'],[False])
            self.assertEqual(action['discrete_action'][0],1)
            self.assertEqual(frame['hold_after'],[2])

    def test_observer_does_not_change_actions_and_ignores_other_policy(self):
        p=self.make();q=self.make()
        for policy in (p,q):policy._hold_left=[3];policy._held_action=[(-.5,-1.,1.)]
        obs=self.observation([295.5]*4)
        with stage_taps(lambda:p) as frame:
            a=p.get_action(obs);saved=copy.deepcopy(frame);b=q.get_action(obs)
            self.assertEqual(frame,saved)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_heat_airflow_override_is_not_cooling_hold(self):
        p=self.make()
        with stage_taps(lambda:p) as frame:
            action=p.get_action(self.observation([293.,295.,295.,297.]))
            self.assertEqual(frame['parent']['run'],[0])
            self.assertEqual(frame['hold'],[False]);self.assertEqual(frame['heat_override'],[True])
            self.assertEqual(action['discrete_action'][0],1)

    def test_aggregate_conserves_handler_runtime_and_integrates_meter_watts(self):
        row=dict(phase='offhours',dt_seconds=300.,run=[1,1],parent_run=[0,1],hold=[True,False],
            heat_override=[False,False],pump=1,open_valves=0,held_reheat=[False,False],starts=[1,0],pump_start=1,
            refreshed=[False,True],new_hold=[False,False],hold_origin=['occupied','offhours'],
            watts=dict(blower=600.,cooling=0.,pump=1200.,gas=0.,heatpump=0.))
        out=aggregate([row])['offhours']
        self.assertAlmostEqual(out['ahu_handler_hours'],2/12)
        self.assertAlmostEqual(out['hold_handler_hours'],1/12)
        self.assertAlmostEqual(out['pump_hours'],1/12)
        self.assertEqual(out['metered_Wh']['blower'],50.)
        self.assertEqual(out['metered_Wh']['pump'],100.)
        self.assertEqual(out['hold_origin_phase_steps']['occupied'],1)
        broken=copy.deepcopy(row);broken['parent_run']=[1,1]
        with self.assertRaises(ValueError):aggregate([broken])

    def test_extra_hold_and_removed_runs_are_not_hidden_by_net_runtime(self):
        from hvaclab.plant_runtime_review import compare_pair
        def row(t,on,held):
            return dict(step=t,phase='offhours',dt_seconds=300.,run=[on],parent_run=[int(on and not held)],
                parent_sat=[1.],hold=[held],hold_origin=['occupied' if held else None],heat_override=[False],
                pump=0,open_valves=0,held_reheat=[False],starts=[on],pump_start=0,refreshed=[False],new_hold=[False],
                watts={k:0. for k in ('blower','cooling','pump','gas','heatpump')},weighted_energy=0.,weighted_carbon=0.)
        a=[row(0,0,False),row(1,1,False)];b=[row(0,1,True),row(1,0,False)]
        def record(series):return dict(result={'steps':2},plant_telemetry=dict(series=series,phase_totals=aggregate(series),ahu_ids=['a'],transitions=[]))
        result=compare_pair(record(a),record(b))
        self.assertEqual(sum(result['offhours_extra_on_handler_steps']),1)
        self.assertEqual(sum(result['offhours_removed_on_handler_steps']),1)
        self.assertEqual(result['offhours_extra_on_reason']['hold'],1)
        self.assertEqual(result['offhours_extra_on_hold_origin']['occupied'],1)
        self.assertAlmostEqual(result['phase_comparison']['offhours']['delta']['ahu_handler_hours'],0.)

    def test_reward_meter_components_are_extracted_without_repricing(self):
        info=RewardInfo()
        info.air_handler_reward_infos['a'].blower_electrical_energy_rate=100.
        info.air_handler_reward_infos['a'].air_conditioning_electrical_energy_rate=200.
        info.boiler_reward_infos['b'].pump_electrical_energy_rate=50.
        info.boiler_reward_infos['b'].natural_gas_heating_energy_rate=1000.
        info.heat_pump_reward_infos['h'].pump_electrical_energy_rate=70.
        info.heat_pump_reward_infos['h'].electricity_heating_energy_rate=2000.
        self.assertEqual(metered_watts(info),dict(blower=100.,cooling=200.,pump=120.,gas=1000.,heatpump=2000.))


if __name__=='__main__':unittest.main()
