"""Fresh-run identity, clock and audit gates; no simulation or generation."""
import copy
import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'reports/scripts/run_fresh_transfer.py'
s=importlib.util.spec_from_file_location('fresh_transfer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class FreshTransferTest(unittest.TestCase):
    def fixture(self):
        task=dict(building='synthetic_1',weather='denver_4',policy='V10',steps=8640)
        d=dict(status='complete_verified',task=task,protocol_sha256='p',cohort_sha256='c',
            sums=dict(steps=8640,reward=-8640.,comfort=4320.,energy=2160.,carbon=2160.),
            means=dict(loss=1.,comfort=.5,energy=.25,carbon=.25),shadow_action_parity_steps=8640,
            morning_comfort={'day':dict(steps=24,first_occupied_epoch=1000.,prod_regret=-1.)},
            allocation_audit=dict(max_relative_actual_flow_error=0.,max_relative_cooling_error=0.,max_supply_error_K=0.))
        return task,d
    def test_full_unique_grid_required(self):
        buildings=[f'synthetic_{i}' for i in range(12)]
        self.assertEqual(len(m.expected_tasks(buildings)),72)
        with self.assertRaises(ValueError):m.expected_tasks(buildings[:-1])
        with self.assertRaises(ValueError):m.expected_tasks(buildings[:-1]+buildings[:1])
    def test_identity_short_and_physical_audit_rejected(self):
        task,d=self.fixture();self.assertTrue(m.valid_result(d,task,'p','c'))
        for edit in ('fingerprint','steps','flow','supply','parity','nan_audit'):
            x=copy.deepcopy(d)
            if edit=='fingerprint':x['cohort_sha256']='changed'
            elif edit=='steps':x['sums']['steps']=8639
            elif edit=='flow':x['allocation_audit']['max_relative_actual_flow_error']=3e-6
            elif edit=='supply':x['allocation_audit']['max_supply_error_K']=.011
            elif edit=='parity':x['shadow_action_parity_steps']=8639
            else:x['allocation_audit']['max_supply_error_K']=float('nan')
            with self.assertRaises(ValueError):m.valid_result(x,task,'p','c')
    def test_mean_decomposition_and_nonfinite_rejected(self):
        task,d=self.fixture()
        for field in ('loss','comfort'):
            x=copy.deepcopy(d);x['means'][field]+=.01
            with self.assertRaises(ValueError):m.valid_result(x,task,'p','c')
        x=copy.deepcopy(d);x['sums']['reward']=float('nan')
        with self.assertRaises(ValueError):m.valid_result(x,task,'p','c')
    def test_morning_identity_and_original_tolerance(self):
        base={'day':dict(steps=24,first_occupied_epoch=1000.,prod_regret=0.)}
        small=copy.deepcopy(base);small['day']['prod_regret']=-24e-7/.6
        self.assertEqual(m.morning_compare(base,small)['failed_mornings'],[])
        worse=copy.deepcopy(base);worse['day']['prod_regret']=-24e-5/.6
        self.assertEqual(len(m.morning_compare(base,worse)['failed_mornings']),1)
        worse['day']['first_occupied_epoch']+=300
        with self.assertRaises(ValueError):m.morning_compare(base,worse)
        with self.assertRaises(ValueError):m.morning_compare(base,{})
    def test_weather_inventory_ignores_hash_keys(self):
        content=dict(source_sha256={'weather':'denver_4'},tasks=[dict(weather='singapore_5')])
        self.assertEqual(m.collect_weather(content,set(m.WEATHER)),{'singapore_5'})
    def test_smoke_short_steps_are_only_accepted_explicitly(self):
        task,d=self.fixture();task['steps']=864;d['sums']=dict(steps=864,reward=-864.,comfort=432.,energy=216.,carbon=216.);d['shadow_action_parity_steps']=864
        with self.assertRaises(ValueError):m.valid_result(d,task,'p','c')
        self.assertTrue(m.valid_result(d,task,'p','c',expected_steps=864))
    def test_generate_marker_requires_active_all_three_arm_smoke(self):
        marker=dict(status='complete_verified_active_integration',locked_protocol_sha256='p',smoke_protocol_sha256='s',jobs=3,steps=864,allocation_checked_zone_steps=7,reserve_zone_steps=3)
        m.validate_smoke_marker(marker,'p','s')
        for key,value in (('jobs',2),('steps',863),('allocation_checked_zone_steps',0),('reserve_zone_steps',0),('locked_protocol_sha256','different')):
            changed=dict(marker);changed[key]=value
            with self.assertRaises(ValueError):m.validate_smoke_marker(changed,'p','s')

if __name__=='__main__':unittest.main()
