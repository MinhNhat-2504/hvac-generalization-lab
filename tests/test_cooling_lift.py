import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.policies.cooling_lift import CoolingLiftPolicy
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata
from test_heat_delivery import observation


def meta():
    m=metadata();m['action_normalization_metadata'].update({
        '1_supply_air_static_pressure_setpoint':dict(min_native_value=0.,max_native_value=2000.,min_normalized_value=-1.,max_normalized_value=1.),
        '1_supply_air_temperature_setpoint':dict(min_native_value=285.,max_native_value=305.,min_normalized_value=-1.,max_normalized_value=1.)})
    m['observation_field_names']+=['ahs_1_supply_air_flowrate_sensor','ahs_1_supply_fan_speed_percentage_command']+[f'VAV_{i}_supply_air_flowrate_setpoint' for i in range(4)]
    return m


class CoolingLiftTest(unittest.TestCase):
    def make(self,enabled=True,**kw):
        register();return CoolingLiftPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=False,cooling_lift=enabled,**kw)
    def obs(self):return np.r_[observation([297.]*4),298.,1000.,.7,310.,4.,.2,[2.]*4]
    def command(self):
        c=np.full(13,-1.,dtype=np.float32);c[0]=-.5;c[1]=0.;c[[5,7,9,11]]=0.
        return dict(continuous_action=c,discrete_action=np.array([1,0]))

    def test_estimated_cooling_preserved_while_supply_and_flow_are_reduced(self):
        p=self.make();x=self.obs();saved=x.copy()
        with patch.object(FlowEquivalentPolicy,'get_action',return_value=self.command()):a=p.get_action(x)
        allocation=p.last_allocations['1'];self.assertLess(allocation['new_supply'],allocation['old_supply'])
        for t,q,new in zip(allocation['observed_temperature'],allocation['old_flows'],allocation['realized_flows']):
            self.assertAlmostEqual(q*(t-allocation['old_supply']),new*(t-allocation['new_supply']),places=5)
            self.assertLess(new,q)
        self.assertEqual(p.lift_handler_steps,1);np.testing.assert_array_equal(x,saved)

    def test_cold_room_reheat_missing_capacity_and_floor_constraints_veto(self):
        for mode in ('cold','outside','reheat','capacity','saturated','floor','off'):
            p=self.make();x=self.obs();command=self.command()
            if mode=='cold':x[6]=294.
            if mode=='outside':x[13]=270.
            if mode=='reheat':command['continuous_action'][6]=0.
            if mode=='capacity':x[-6]=0.
            if mode=='saturated':x[-6]=.2
            if mode=='floor':
                command['continuous_action'][5]=-1.;command['continuous_action'][1]=-.8
            if mode=='off':command['discrete_action'][0]=0
            saved={k:v.copy() for k,v in command.items()}
            with patch.object(FlowEquivalentPolicy,'get_action',return_value=command):a=p.get_action(x)
            for k in a:np.testing.assert_array_equal(a[k],saved[k])
            self.assertFalse(p.last_allocations)

    def test_disabled_matches_existing_baseline_and_reset_clears_diagnostics(self):
        p=self.make(False);q=FlowEquivalentPolicy(meta(),None,**_VARIANTS['damper12_mid'][1],flow_equivalent=False)
        for _ in range(40):
            a,b=p.get_action(self.obs()),q.get_action(self.obs())
            for k in a:np.testing.assert_array_equal(a[k],b[k])
        p=self.make()
        with patch.object(FlowEquivalentPolicy,'get_action',return_value=self.command()):p.get_action(self.obs())
        p.reset();self.assertEqual(p.lift_handler_steps,0);self.assertFalse(p.last_allocations)
        for value in (0.,3.,float('nan')):
            with self.assertRaises(ValueError):self.make(supply_reduction_k=value)


if __name__=='__main__':unittest.main()
