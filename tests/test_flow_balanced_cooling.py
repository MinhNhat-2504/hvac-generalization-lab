import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from hvaclab.submission.experimental import assemble_variant
from test_heat_delivery import building
from test_cooling_pressure import obs


def make(key):
    meta=building()
    meta["observation_field_names"].append("ahs_1_outside_air_temperature_sensor")
    return resolve_policy(key)(meta,None)


def prime(policy):
    policy.get_action(obs([297.01]*4))
    policy._zone_integral.update(VAV_0=170.,VAV_1=80.,VAV_2=-80.,VAV_3=0.)


class FlowBalancedCoolingTest(unittest.TestCase):
    def test_nominal_non_target_flow_is_preserved_above_damper_floor(self):
        baseline,candidate=make("e_sync"),make("cfb_comp")
        for p in (baseline,candidate):
            prime(p)
        a,b=baseline.get_action(obs([298,295.5,295.5,295.5])),candidate.get_action(obs([298,295.5,295.5,295.5]))
        self.assertAlmostEqual(float(b["continuous_action"][1]),-.6,places=6)
        old_p=max(.1,.5*(float(a["continuous_action"][1])+1))
        new_p=max(.1,.5*(float(b["continuous_action"][1])+1))
        for index in (7,9):
            old_d=max(.1,.5*(float(a["continuous_action"][index])+1))
            new_d=max(.1,.5*(float(b["continuous_action"][index])+1))
            self.assertAlmostEqual(old_d*np.sqrt(old_p),new_d*np.sqrt(new_p),places=7)
        self.assertEqual(a["continuous_action"][5],b["continuous_action"][5])
        self.assertEqual(a["continuous_action"][10],b["continuous_action"][10])
        np.testing.assert_array_equal(a["discrete_action"],b["discrete_action"])

    def test_cold_minimum_flow_neighbor_vetoes_entire_handler_change(self):
        baseline,candidate=make("e_sync"),make("cfb_guard")
        for p in (baseline,candidate):
            prime(p)
        observation=obs([298,295.5,295.5,294.4])
        a,b=baseline.get_action(observation),candidate.get_action(observation)
        for field in a:
            np.testing.assert_array_equal(a[field],b[field])
        self.assertGreater(candidate.flow_guard_vetoes,0)

    def test_disabled_control_matches_v4(self):
        baseline,candidate=make("e_sync"),make("cfb_ctrl")
        for t in ([297.01]*4,[298,295.5,295.5,295.5],[292]*4)*100:
            a,b=baseline.get_action(obs(t)),candidate.get_action(obs(t))
            for field in a:
                np.testing.assert_array_equal(a[field],b[field])

    def test_standalone_matches_compensation_and_veto_paths(self):
        for key in ("cfb_comp","cfb_guard"):
            source,_=assemble_variant(key)
            namespace={}
            exec(compile(source,"flow_candidate.py","exec"),namespace)
            meta=building()
            meta["observation_field_names"].append("ahs_1_outside_air_temperature_sensor")
            standalone=namespace["FinalPolicy"](meta,None)
            reference=make(key)
            for p in (standalone,reference):
                prime(p)
            for t in ([298,295.5,295.5,295.5],[298,295.5,295.5,294.4]):
                a,b=standalone.get_action(obs(t)),reference.get_action(obs(t))
                for field in a:
                    np.testing.assert_array_equal(a[field],b[field])


if __name__ == "__main__":
    unittest.main()
