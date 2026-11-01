import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.policies.comfort_hold_release import ComfortHoldReleasePolicy
from hvaclab.policies.zone_pi import ZonePIPolicy
from hvaclab.registry import resolve_policy
from test_heat_delivery import building,observation


class HoldReleaseTest(unittest.TestCase):
    def make(self,**kwargs):
        return ComfortHoldReleasePolicy(building(),None,reheat_exponent=.75,
                                        hold_steps=288,coordinate_water=True,
                                        setback_k=0.,setup_k=0.,heating_pressure=-1.,
                                        cooling_pressure_gain=1.,reheat_full_k=.5,**kwargs)

    def arm(self,policy):
        policy._hold_left=[100]
        policy._held_action=[(-.5,-1.,1.)]

    def test_disabled_control_matches_v7(self):
        left=self.make(release_enabled=False)
        right=resolve_policy('valve8_gentle')(building(),None)
        for temps in ([299]*4,[293,295,295,295],[295.5]*4,[292]*4)*50:
            obs=observation(temps)
            a,b=left.get_action(obs),right.get_action(obs)
            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_release_requires_three_consecutive_cold_steps_and_clears_only_hold(self):
        p=self.make()
        self.arm(p)
        obs=observation([293,295,295,295])
        # Freeze only the underlying parent request to idle. This isolates
        # hold expiration without bypassing the hold/heat-delivery layers.
        def idle(_self,obs):
            p._pi_active=True
            p._zone_u={f'VAV_{i}':0. for i in range(4)}
            return dict(continuous_action=-np.ones(13,dtype=np.float32),
                        discrete_action=np.zeros(2,dtype=int))
        with patch.object(ZonePIPolicy,'get_action',idle):
            self.assertEqual(p.get_action(obs)['discrete_action'][0],1)
            self.assertEqual(p.get_action(obs)['discrete_action'][0],1)
            self.assertEqual(p.get_action(obs)['discrete_action'][0],0)
        self.assertEqual(p.holds_released,1)
        self.assertEqual(p._hold_left,[0])

    def test_hot_reserve_inactive_and_nonfinite_readings_veto_release(self):
        for obs in (observation([293,296,295,295]),
                    observation([293,295,295,295],hour=23,comfort=False),
                    observation([293,float('nan'),295,295])):
            p=self.make()
            self.arm(p)
            with patch.object(p.__class__.__bases__[0],'get_action',return_value={}):
                for _ in range(5):p.get_action(obs)
            self.assertEqual(p.holds_released,0)
            self.assertEqual(p._cold_streak,[0])

    def test_new_parent_cooling_request_is_not_suppressed(self):
        p=self.make(confirmation_steps=1)
        self.arm(p)
        def cooling(_self,obs):
            p._pi_active=True
            p._zone_u={f'VAV_{i}':0. for i in range(4)}
            cont=-np.ones(13,dtype=np.float32)
            cont[0]=-.5
            return dict(continuous_action=cont,discrete_action=np.array([1,0]))
        with patch.object(ZonePIPolicy,'get_action',cooling):
            action=p.get_action(observation([293,295,295,295]))
        self.assertEqual(action['discrete_action'][0],1)
        self.assertEqual(p._hold_left,[288])

    def test_warm_interruption_resets_cold_confirmation(self):
        p=self.make()
        self.arm(p)
        cold=observation([293,295,295,295])
        warm=observation([293,296,295,295])
        with patch.object(p.__class__.__bases__[0],'get_action',return_value={}):
            p.get_action(cold)
            p.get_action(cold)
            p.get_action(warm)
            p.get_action(cold)
            self.assertEqual(p._cold_streak,[1])
            self.assertEqual(p.holds_released,0)

    def test_new_parent_heating_request_is_preserved(self):
        p=self.make(confirmation_steps=1)
        self.arm(p)
        obs=observation([293,295,295,295])
        # Force only the aggregate's legitimate new request. The hold-release
        # extension must not rewrite it even if the release condition fires.
        def heating(_self,obs):
            p._pi_active=True
            p._zone_u={f'VAV_{i}':-.5 for i in range(4)}
            cont=-np.ones(13,dtype=np.float32)
            cont[0]=1.
            return dict(continuous_action=cont,discrete_action=np.array([1,1]))
        with patch.object(ZonePIPolicy,'get_action',heating):
            action=p.get_action(obs)
        self.assertEqual(action['discrete_action'][0],1)
        self.assertEqual(float(action['continuous_action'][0]),1.)

    def test_invalid_parameters_rejected(self):
        for args in ({'cold_deficit_k':0.},{'cold_deficit_k':float('nan')},
                     {'confirmation_steps':0},{'confirmation_steps':1.5}):
            with self.assertRaises(ValueError):self.make(**args)


if __name__=='__main__':unittest.main()
