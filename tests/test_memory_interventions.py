"""Interventions edit only the declared controller memories."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import unittest

p=Path(__file__).resolve().parents[1]/'reports/scripts/isolate_controller_memory.py'
s=importlib.util.spec_from_file_location('memory',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class MemoryInterventionTest(unittest.TestCase):
    def make(self):return SimpleNamespace(_zone_integral={'v':3.},_zone_u={'v':.4},_hold_left=[8,9],_held_action=[(1,2,3),(4,5,6)],_previous_lift_temperatures={'v':297.},zone_kp=.1)
    def test_pi_edit_preserves_hold_projection_and_signed_output(self):
        p=self.make();m.intervene(p,True,False)
        self.assertEqual(p._zone_integral,{});self.assertEqual(p._hold_left,[8,9]);self.assertEqual(p._zone_u,{'v':.4});self.assertEqual(p._previous_lift_temperatures,{'v':297.})
    def test_hold_edit_preserves_pi_and_clears_cached_commands(self):
        p=self.make();m.intervene(p,False,True)
        self.assertEqual(p._zone_integral,{'v':3.});self.assertEqual(p._hold_left,[0,0]);self.assertEqual(p._held_action,[None,None]);self.assertEqual(p.zone_kp,.1)
    def test_sham_preserves_all_fields(self):
        p=self.make();before=p.__dict__.copy();m.intervene(p);self.assertEqual(p.__dict__,before)
if __name__=='__main__':unittest.main()
