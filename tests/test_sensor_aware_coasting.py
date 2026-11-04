import unittest
import numpy as np
from hvaclab.policies.sensor_aware_coasting import SensorAwareCoastingPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS,resolve_policy
from test_heat_delivery import building,observation


class SensorAwareTest(unittest.TestCase):
    def make(self,**kwargs):
        register()
        return SensorAwareCoastingPolicy(building(),None,**{**_VARIANTS['damper12_mid'][1],
            'coast_mode':'all','minimum_on_steps':1,'forecast_steps':1,**kwargs})

    def arm(self,p):
        p._hold_left=[100]
        p._held_action=[(-.5,-1.,1.)]

    def test_release_commands_ahu_on_and_holds_recovery(self):
        p=self.make(recovery_steps=3)
        self.arm(p)
        for _ in range(2):p.get_action(observation([295.5]*4))
        self.assertTrue(p._coasting[0])
        p._on_steps[0]=0
        a=p.get_action(observation([296.4]*4))
        self.assertFalse(p._coasting[0])
        self.assertEqual(a['discrete_action'][0],1)
        self.assertEqual(p.guard_releases,1)
        for _ in range(2):
            a=p.get_action(observation([295.5]*4))
            self.assertEqual(a['discrete_action'][0],1)
            self.assertFalse(p._coasting[0])
        self.assertEqual(p.recovery_run_steps,3)

    def test_age_correction_rejects_late_entry(self):
        a=self.make(sensor_age_steps=0)
        b=self.make(sensor_age_steps=1)
        for p in (a,b):
            self.arm(p)
            p.get_action(observation([295.9]*4))
        # Current+1 slope remains within 296.25, current+2 does not.
        a.get_action(observation([296.05]*4));b.get_action(observation([296.05]*4))
        self.assertTrue(a._coasting[0])
        self.assertFalse(b._coasting[0])

    def test_disabled_matches_v10(self):
        p=self.make(coast_mode='off');q=resolve_policy('damper12_mid')(building(),None)
        for temperature in [293.,295.5,299.,296.]*8:
            obs=observation([temperature]*4)
            a,b=p.get_action(obs),q.get_action(obs)
            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_heating_recovery_keeps_water_and_warm_sat(self):
        p=self.make(recovery_steps=2)
        self.arm(p)
        for _ in range(2):p.get_action(observation([295.5]*4))
        self.assertTrue(p._coasting[0])
        action=p.get_action(observation([293.]*4))
        self.assertEqual(action['discrete_action'][0],1)
        self.assertEqual(action['discrete_action'][p.water_discrete_idx],1)
        self.assertGreaterEqual(action['continuous_action'][p.ahu_controllers[0]['temp_idx']],0.)

    def test_invalid_recovery_rejected(self):
        for value in (0,True,1.5):
            with self.assertRaises(ValueError):self.make(recovery_steps=value)


if __name__=='__main__':unittest.main()
