from contextlib import redirect_stdout
import io
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from hvaclab.policies.gated_reheat import GatedReheatPolicy
from hvaclab.registry import resolve_policy
from hvaclab.submission.build import build_notebook
from hvaclab.submission.experimental import assemble_variant,notebook_template,GATED_REHEAT
from test_heat_delivery import building,observation


class GatedReheatTest(unittest.TestCase):
    def test_off_and_all_controls_reproduce_existing_controllers(self):
        for left_key,right_key in (("valve9_ctrl","e_sync"),("valve9_curve_ctrl","valve8_gentle")):
            left,right=resolve_policy(left_key)(building(),None),resolve_policy(right_key)(building(),None)
            for hour,temps in [(12,[292,296,296,296]),(12,[293,299,299,299]),
                               (12,[299]*4),(23,[292]*4),(5,[294]*4)]*100:
                obs=observation(temps,hour,comfort=hour==12)
                a,b=left.get_action(obs),right.get_action(obs)
                for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_heating_gate_uses_final_same_step_command(self):
        for temps,heating in (([292,296,296,296],True),([293,299,299,299],False)):
            policy=resolve_policy("valve9_heat")(building(),None)
            action=policy.get_action(observation(temps))
            self.assertEqual(float(action['continuous_action'][0])>0,heating)
            h=-policy._zone_u['VAV_0']
            self.assertGreater(h,0)
            opening=h**.75 if heating else h
            self.assertAlmostEqual(float(action['continuous_action'][6]),2*opening-1,places=6)

    def test_cold_blend_is_zero_at_band_edge_and_full_half_kelvin_below(self):
        for temp,weight in ((294.25,0.),(294.,0.),(293.75,.5),(293.5,1.),(293.,1.)):
            policy=resolve_policy('valve9_cold')(building(),None)
            action=policy.get_action(observation([temp,299,299,299]))
            h=max(0.,-policy._zone_u['VAV_0'])
            expected=h+weight*(h**.75-h)
            self.assertAlmostEqual(float(action['continuous_action'][6]),2*expected-1,places=6)

    def test_only_valves_change_and_inactive_schedule_is_untouched(self):
        for key in GATED_REHEAT:
            left,right=resolve_policy(key)(building(),None),resolve_policy('e_sync')(building(),None)
            for hour,temps in [(12,[293.75,295,296,299]),(12,[292]*4),
                               (12,[299]*4),(23,[292]*4)]*30:
                obs=observation(temps,hour,comfort=hour==12)
                a,b=left.get_action(obs),right.get_action(obs)
                np.testing.assert_array_equal(a['discrete_action'],b['discrete_action'])
                mask=np.ones(len(a['continuous_action']),dtype=bool)
                if hour==12:mask[[6,8,10,12]]=False
                np.testing.assert_array_equal(a['continuous_action'][mask],b['continuous_action'][mask])
                self.assertEqual(left._zone_u,right._zone_u)
                self.assertEqual(left._zone_integral,right._zone_integral)

    def test_full_notebook_wrapper_matches_registered_gates_and_logs_variant(self):
        for key in GATED_REHEAT:
            source,_=assemble_variant(key)
            notebook=build_notebook(source,'','','',template=notebook_template(source,key))
            code=''.join(notebook['cells'][0]['source'])
            captured={}
            class Base:
                def __init__(self,*args,**kwargs):pass
            class Server:
                def __init__(self,policy):captured['policy']=policy
                def run(self,**kwargs):pass
            module=types.ModuleType('kaggle_evaluation')
            module.smart_buildings_inference_server=types.SimpleNamespace(Policy=Base,SmartBuildingsInferenceServer=Server)
            output=io.StringIO()
            saved_path=list(sys.path)
            try:
                with patch.dict(sys.modules,{'kaggle_evaluation':module}),redirect_stdout(output):
                    exec(compile(code,'gated_notebook.py','exec'),{})
            finally:sys.path[:]=saved_path
            self.assertIn(f'variant={key}',output.getvalue())
            left=captured['policy'](building(),None)
            right=resolve_policy(key)(building(),None)
            for temps in ([292,296,296,296],[293.75,299,299,299],[299]*4)*30:
                obs=observation(temps)
                a,b=left.get_action(obs),right.get_action(obs)
                for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_zero_and_full_valve_authority_are_preserved(self):
        for key in GATED_REHEAT:
            cold=resolve_policy(key)(building(),None)
            cold._zone_integral={f'VAV_{i}':-200. for i in range(4)}
            action=cold.get_action(observation([292]*4))
            np.testing.assert_array_equal(action['continuous_action'][[6,8,10,12]],np.ones(4))
            neutral=resolve_policy(key)(building(),None)
            action=neutral.get_action(observation([295.5]*4))
            np.testing.assert_array_equal(action['continuous_action'][[6,8,10,12]],-np.ones(4))

    def test_invalid_gate_and_blend_parameters_rejected(self):
        for kwargs in ({'curve_mode':'bad'},{'cold_full_k':0.},{'cold_full_k':float('nan')},
                       {'reheat_exponent':float('inf')},{'reheat_exponent':-1.}):
            with self.assertRaises(ValueError):GatedReheatPolicy(building(),None,**kwargs)


if __name__=='__main__':unittest.main()
