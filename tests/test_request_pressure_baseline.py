"""Semantic tests for pressure-only requests, lifecycle and bounded public actions."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.policies.offhours_hold import OffhoursHoldPolicy
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
import test_cooling_lift as fixture

SCRIPT=Path(__file__).resolve().parents[1]/'reports/scripts/request_pressure_baseline.py'
spec=importlib.util.spec_from_file_location('request_pressure',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RequestPressureTest(unittest.TestCase):
    def make(self,metadata=None,**kwargs):
        register();return module.RequestPressureBaseline(metadata or fixture.meta(),None,
            **_VARIANTS['damper12_mid'][1],**kwargs)

    def command(self,opening=.5,running=True,pressure=1000.):
        c=fixture.CoolingLiftTest().command();c['continuous_action'][1]=pressure/1000.-1.
        c['continuous_action'][[5,7,9,11]]=2*opening-1
        c['discrete_action'][0]=int(running);return c

    def step(self,p,**kwargs):
        command=self.command(**kwargs)
        with patch.object(OffhoursHoldPolicy,'get_action',return_value=command):result=p.get_action(np.zeros(20))
        return result,command

    def native(self,action):return (float(action['continuous_action'][1])+1)*1000

    def test_startup_then_trim_preserves_other_commands(self):
        p=self.make();a,_=self.step(p);self.assertAlmostEqual(self.native(a),1000.,places=3)
        for _ in range(3):a,_=self.step(p)
        self.assertAlmostEqual(self.native(a),970.,places=3)
        np.testing.assert_array_equal(a['continuous_action'][[0,2,3,4,5,6,7,8,9,10,11,12]],p.last_parent_action['continuous_action'][[0,2,3,4,5,6,7,8,9,10,11,12]])
        np.testing.assert_array_equal(a['discrete_action'],p.last_parent_action['discrete_action'])

    def test_request_hysteresis_and_maximum_response(self):
        p=self.make();self.step(p,opening=.96)
        a,_=self.step(p,opening=.90);self.assertAlmostEqual(self.native(a),1037.,places=3)
        self.assertEqual(p.last_request_counts,[4])
        a,_=self.step(p,opening=.84);self.assertAlmostEqual(self.native(a),1027.,places=3)
        self.assertEqual(p.last_request_counts,[0])

    def test_ignored_requests_and_update_cadence(self):
        p=self.make(ignored_requests=4,update_steps=2,startup_delay_steps=2)
        pressures=[]
        for _ in range(6):a,_=self.step(p,opening=.98);pressures.append(round(self.native(a),3))
        self.assertEqual(pressures,[1000.,1000.,990.,990.,980.,980.])

    def test_off_resets_request_pressure_and_restart_uses_inherited(self):
        p=self.make();self.step(p,opening=.96);self.step(p,opening=.96)
        off,_=self.step(p,running=False);self.assertIsNone(p._request_pressure[0])
        self.assertEqual(p._zone_requests,[[False]*4]);self.assertEqual(float(off['continuous_action'][1]),0.)
        a,_=self.step(p,opening=.5,pressure=600.);self.assertAlmostEqual(self.native(a),600.,places=3)
        p.reset();self.assertIsNone(p._request_pressure[0]);self.assertEqual(p.respond_updates,0)

    def test_pressure_floor_ceiling_and_missing_metadata(self):
        p=self.make();self.step(p,pressure=200.)
        for _ in range(3):a,_=self.step(p,pressure=200.)
        self.assertAlmostEqual(self.native(a),200.,places=3)
        p=self.make();self.step(p,pressure=2000.,opening=1.)
        a,_=self.step(p,pressure=2000.,opening=1.);self.assertAlmostEqual(self.native(a),2000.,places=3)
        m=fixture.meta();m['action_normalization_metadata'].pop('1_supply_air_static_pressure_setpoint')
        p=self.make(m);a,_=self.step(p);np.testing.assert_array_equal(a['continuous_action'],p.last_parent_action['continuous_action'])
        self.assertEqual(p.unsupported_handler_steps,1)

    def test_disabled_matches_parent_over_history(self):
        p=self.make(request_reset=False);q=OffhoursHoldPolicy(fixture.meta(),None,**_VARIANTS['damper12_mid'][1])
        obs=fixture.CoolingLiftTest().obs()
        for _ in range(24):
            a=p.get_action(obs);b=q.get_action(obs)
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_invalid_configuration_and_nonfinite_command_fall_back(self):
        for kwargs in ({'trim_pa':0},{'respond_pa':float('nan')},{'request_off':.99},{'update_steps':True},{'startup_delay_steps':-1}):
            with self.assertRaises(ValueError):self.make(**kwargs)
        p=self.make();c=self.command();c['continuous_action'][5]=np.nan
        with patch.object(OffhoursHoldPolicy,'get_action',return_value=c):a=p.get_action(np.zeros(20))
        self.assertTrue(np.isnan(a['continuous_action'][5]));self.assertIsNone(p._request_pressure[0])


if __name__=='__main__':unittest.main()
