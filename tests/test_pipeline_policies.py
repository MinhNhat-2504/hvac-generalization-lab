import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from hvaclab.pipeline_entry import register,assemble,SPECS
from hvaclab.pipeline_queue import dependencies_ready
from hvaclab.registry import resolve_policy
from test_heat_delivery import building,observation
from test_coverage import workspace_directory


class PipelinePolicyTest(unittest.TestCase):
    def test_packaging_exports_the_correct_source_and_marker_for_both_families(self):
        from hvaclab import pipeline_entry as entry
        from hvaclab.submission import experimental as bundle
        entry.register()
        original_root=bundle.REPO
        for key in ('curve10_mid','damper11_less','damper12_mid','persist13_long'):
            expected,_=entry.assemble(key)
            with workspace_directory() as root:
                class RoutedRoot:
                    def __truediv__(self,part):
                        return root/part if part in ('experiments','experiments/submission') else original_root/part
                    def __fspath__(self):return str(original_root)
                # Restore every child-local adapter when this test finishes.
                with patch.multiple(bundle,REPO=RoutedRoot(),SUPPORTED=bundle.SUPPORTED,
                                    REHEAT_CURVE=bundle.REHEAT_CURVE,
                                    variant_sources=bundle.variant_sources,
                                    assemble_variant=bundle.assemble_variant), \
                     patch.object(sys,'argv',['entry','--package','--variant',key]), \
                     redirect_stdout(io.StringIO()):
                    entry.package()
                out=root/'experiments/submission'/key
                self.assertEqual((out/'submission_policy.py').read_text(encoding='utf-8'),expected)
                manifest=json.loads((out/'manifest.json').read_text())
                self.assertEqual(manifest['variant'],key)
                notebook=json.loads((out/'submission.ipynb').read_text())
                code=''.join(notebook['cells'][0]['source'])
                compile(code,key,'exec')
                self.assertIn('HVAC_SUBMISSION variant='+key,code)
                self.assertIn(manifest['runtime_source_lf_sha256'],code)

    def test_queue_requires_every_dependency_complete_and_no_live_scheduler(self):
        states={'one':dict(status='complete'),'two':dict(status='running')}
        self.assertFalse(dependencies_ready(states,set()))
        states['two']['status']='complete'
        self.assertFalse(dependencies_ready(states,{'hvaclab.one'}))
        self.assertTrue(dependencies_ready(states,set()))

    def test_each_control_reproduces_persist432_and_all_standalones_match(self):
        register()
        for study,spec in SPECS.items():
            for key in spec['arms']:
                source,_=assemble(key)
                scope={}
                exec(compile(source,key,'exec'),scope)
                left=resolve_policy(key)(building(),None)
                right=scope['FinalPolicy'](building(),None)
                baseline=resolve_policy(spec.get('base_key','persist_432'))(building(),None) if key==spec['control'] else None
                for hour,temp in [(12,[292,295,296,299]),(12,[299]*4),(12,[295.5]*4),
                                  (23,[293]*4),(5,[292]*4)]*40:
                    obs=observation(temp,hour,comfort=hour==12)
                    a=left.get_action(obs)
                    for other in (right,baseline):
                        if other is not None:
                            b=other.get_action(obs)
                            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_damper_arms_change_only_active_heating_dampers_and_keep_pi_valves(self):
        register()
        for key in SPECS['heating_damper']['arms']:
            left=resolve_policy(key)(building(),None)
            right=resolve_policy('persist_432')(building(),None)
            for hour,temps in [(12,[294,295,296,299]),(12,[292]*4),(23,[293]*4)]*30:
                obs=observation(temps,hour,comfort=hour==12)
                a,b=left.get_action(obs),right.get_action(obs)
                np.testing.assert_array_equal(a['discrete_action'],b['discrete_action'])
                mask=np.ones(13,dtype=bool)
                if hour==12:
                    for i in range(4):
                        u=left._zone_u[f'VAV_{i}']
                        if u<0:
                            mask[5+2*i]=False
                            expected=2*(-u)**left.heating_damper_exponent-1
                            self.assertAlmostEqual(float(a['continuous_action'][5+2*i]),expected,places=6)
                np.testing.assert_array_equal(a['continuous_action'][mask],b['continuous_action'][mask])
                self.assertEqual(left._zone_integral,right._zone_integral)
                self.assertEqual(left._zone_u,right._zone_u)

    def test_new_studies_retain_the_declared_v9_base(self):
        register()
        for key in SPECS['damper_refinement']['arms']:
            policy=resolve_policy(key)(building(),None)
            self.assertEqual(policy.hold_steps,432)
            self.assertEqual(policy.reheat_exponent,.75)
        for key,params in SPECS['damper_persistence']['arms'].items():
            policy=resolve_policy(key)(building(),None)
            self.assertEqual(policy.hold_steps,params['hold_steps'])
            self.assertEqual(policy.heating_damper_exponent,1.25)
            self.assertEqual(policy.reheat_exponent,.75)


if __name__=='__main__':unittest.main()
