from contextlib import redirect_stdout
import hashlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from hvaclab.policies.reheat_curve import ReheatCurvePolicy
from hvaclab.registry import resolve_policy
from hvaclab.submission.build import build_notebook
from hvaclab.submission.experimental import assemble_variant,notebook_template,REHEAT_CURVE
from test_heat_delivery import building,observation


class ReheatCurveTest(unittest.TestCase):
    def test_linear_control_is_exact_v4_over_varying_demands_and_schedule(self):
        left=resolve_policy("valve8_ctrl")(building(),None)
        right=resolve_policy("e_sync")(building(),None)
        for hour,temps in [(12,[292,296,296,296]),(12,[299]*4),(12,[295.5]*4),
                           (23,[294]*4),(5,[292]*4)]*100:
            obs=observation(temps,hour,comfort=hour==12)
            a,b=left.get_action(obs),right.get_action(obs)
            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_only_active_pi_reheat_valve_changes(self):
        for key in REHEAT_CURVE:
            policy=resolve_policy(key)(building(),None)
            baseline=resolve_policy("e_sync")(building(),None)
            for temps in ([294,295,296,299],[295.5]*4,[292]*4)*20:
                obs=observation(temps)
                action,reference=policy.get_action(obs),baseline.get_action(obs)
                np.testing.assert_array_equal(action['discrete_action'],reference['discrete_action'])
                indices=[]
                for ctrl in policy.ahu_controllers:
                    for vav,_,actions in ctrl['zones']:
                        index=actions['reheat_valve_percentage_command']
                        indices.append(index)
                        h=max(0.,-policy._zone_u[vav])
                        self.assertAlmostEqual(float(action['continuous_action'][index]),
                                               2*h**policy.reheat_exponent-1,places=6)
                mask=np.ones(len(action['continuous_action']),dtype=bool)
                mask[indices]=False
                np.testing.assert_array_equal(action['continuous_action'][mask],reference['continuous_action'][mask])
                self.assertEqual(policy._zone_integral,baseline._zone_integral)
                self.assertEqual(policy._zone_u,baseline._zone_u)

    def test_zero_full_authority_and_inactive_law_preserved(self):
        for key in REHEAT_CURVE:
            for integral,expected in ((0.,-1.),(-200.,1.)):
                policy=resolve_policy(key)(building(),None)
                policy._zone_integral={f'VAV_{i}':integral for i in range(4)}
                action=policy.get_action(observation([295.5]*4))
                self.assertEqual(float(action['continuous_action'][6]),expected)
            left=resolve_policy(key)(building(),None)
            right=resolve_policy('e_sync')(building(),None)
            obs=observation([292,299,296,295],hour=23,comfort=False)
            a,b=left.get_action(obs),right.get_action(obs)
            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_notebook_wrapper_matches_registered_policy_and_prints_source_marker(self):
        for key in REHEAT_CURVE:
            source,_=assemble_variant(key)
            notebook=build_notebook(source,'','','',template=notebook_template(source,key))
            code=''.join(notebook['cells'][0]['source'])
            captured={}
            class Base:
                def __init__(self,*args,**kwargs):pass
            class Server:
                def __init__(self,policy):captured['policy']=policy
                def run(self,**kwargs):captured['run']=kwargs
            module=types.ModuleType('kaggle_evaluation')
            module.smart_buildings_inference_server=types.SimpleNamespace(Policy=Base,SmartBuildingsInferenceServer=Server)
            output=io.StringIO()
            original_path=list(sys.path)
            try:
                with patch.dict(sys.modules,{'kaggle_evaluation':module}),redirect_stdout(output):
                    exec(compile(code,'actual_notebook.py','exec'),{})
            finally:
                sys.path[:]=original_path
            self.assertIn(f'variant={key}',output.getvalue())
            self.assertIn(hashlib.sha256(source.encode('utf-8')).hexdigest(),output.getvalue())
            left=captured['policy'](building(),None)
            right=resolve_policy(key)(building(),None)
            for temps in ([292,296,295,299],[299]*4,[295.5]*4)*30:
                a,b=left.get_action(observation(temps)),right.get_action(observation(temps))
                for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_invalid_curves_rejected(self):
        for exponent in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):ReheatCurvePolicy(building(),None,reheat_exponent=exponent)


if __name__=='__main__':unittest.main()
