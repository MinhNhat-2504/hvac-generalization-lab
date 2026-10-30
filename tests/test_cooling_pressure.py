import unittest

import numpy as np

from hvaclab.registry import resolve_policy
from hvaclab.submission.experimental import assemble_variant
from test_heat_delivery import building, observation


def make(key):
    meta=building()
    meta["observation_field_names"].append("ahs_1_outside_air_temperature_sensor")
    return resolve_policy(key)(meta,None)


def obs(temperatures,outside=310.,hour=12):
    return np.append(observation(temperatures,hour,comfort=hour==12),outside)


class CoolingPressureTest(unittest.TestCase):
    def test_standalone_bundle_matches_the_active_boost_path(self):
        source,_=assemble_variant("cpi_400")
        namespace={}
        exec(compile(source,"cpi_standalone.py","exec"),namespace)
        meta=building()
        meta["observation_field_names"].append("ahs_1_outside_air_temperature_sensor")
        standalone=namespace["FinalPolicy"](meta,None)
        reference=make("cpi_400")
        for p in (standalone,reference):
            self.prime(p)
        for observation in (obs([297.5,295.5,295.5,295.5]),
                            obs([298.5,295.5,295.5,295.5]),
                            obs([298.5,295.5,295.5,295.5],260)):
            a,b=standalone.get_action(observation),reference.get_action(observation)
            for field in a:
                np.testing.assert_array_equal(a[field],b[field])
        self.assertGreater(standalone.pressure_boosts,0)

    def prime(self,policy):
        policy.get_action(obs([297.01]*4))
        policy._zone_integral["VAV_0"]=170.

    def test_disabled_control_matches_reference(self):
        baseline,control=make("e_sync"),make("cpi_ctrl")
        for temps in ([297.01]*4,[298,295.5,295.5,295.5],[292]*4)*100:
            a,b=baseline.get_action(obs(temps)),control.get_action(obs(temps))
            for field in a:
                np.testing.assert_array_equal(a[field],b[field])

    def test_saturated_hot_outlier_changes_only_pressure_with_smooth_response(self):
        baseline,candidate=make("e_sync"),make("cpi_400")
        for p in (baseline,candidate):
            self.prime(p)
        observation=obs([297.5,295.5,295.5,295.5])
        saved=observation.copy()
        a,b=baseline.get_action(observation),candidate.get_action(observation)
        self.assertAlmostEqual(float(b["continuous_action"][1]),-.7,places=6)
        expected=a["continuous_action"].copy()
        expected[1]=-.7
        np.testing.assert_array_equal(expected,b["continuous_action"])
        np.testing.assert_array_equal(a["discrete_action"],b["discrete_action"])
        np.testing.assert_array_equal(observation,saved)

    def test_cold_mixing_vetoes_extra_airflow(self):
        baseline,candidate=make("e_sync"),make("cpi_600")
        for p in (baseline,candidate):
            self.prime(p)
        a,b=baseline.get_action(obs([298,295.5,295.5,295.5],260)),candidate.get_action(obs([298,295.5,295.5,295.5],260))
        for field in a:
            np.testing.assert_array_equal(a[field],b[field])

    def test_never_reduces_an_existing_higher_pressure(self):
        baseline,candidate=make("e_sync"),make("cpi_400")
        for p in (baseline,candidate):
            p._zone_integral["VAV_0"]=170.
        a,b=baseline.get_action(obs([299]*4)),candidate.get_action(obs([299]*4))
        for field in a:
            np.testing.assert_array_equal(a[field],b[field])


if __name__ == "__main__":
    unittest.main()
