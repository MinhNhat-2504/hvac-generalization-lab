import unittest

import numpy as np

from hvaclab.policies.split_geometry_pi import SplitGeometryPIPolicy
from hvaclab.registry import resolve_policy
from hvaclab.submission.experimental import assemble_variant
from test_geometry_pi import metadata
from test_heat_delivery import observation


VARIANTS=(("geo7_ctrl",.75,.75),("geo7_soft_i",.75,.5),
          ("geo7_strong_i",.75,1.),("geo7_soft_p",.5,.75))


def realistic_metadata():
    meta=metadata()
    grid=np.ones((102,260))
    column=1
    for side in (25,50,75,100):
        grid[1:side+1,column:column+side]=0
        column+=side+1
    meta["zone_map"]=grid
    return meta


class SplitGeometryTest(unittest.TestCase):
    def test_separate_gains_and_integral_limit_use_each_declared_exponent(self):
        for key,p_exponent,i_exponent in VARIANTS:
            policy=SplitGeometryPIPolicy(metadata(),None,reference_area_cv=4.,
                                         area_exponent=p_exponent,integral_area_exponent=i_exponent)
            policy.get_action(observation([296.5]*4))
            for index,area in enumerate((1,4,9,16),1):
                p_scale=np.clip((area/4.)**p_exponent,.5,2.)
                i_scale=np.clip((area/4.)**i_exponent,.5,2.)
                self.assertAlmostEqual(policy._zone_u[f"VAV_{index}"],.1*p_scale+.005*i_scale)
            self.assertEqual((policy.zone_kp,policy.zone_ki,policy.zone_integral_limit),(.1,.005,200.))
            for zone,scale in policy._zone_integral_gain_scales.items():
                policy._zone_integral[zone]=200./scale
            policy.get_action(observation([295.5]*4))
            for value in policy._zone_u.values():self.assertAlmostEqual(value,1.)

    def test_saturated_integral_stops_and_unwinds(self):
        for key,_,_ in VARIANTS:
            policy=resolve_policy(key)(realistic_metadata(),None)
            for zone,scale in policy._zone_integral_gain_scales.items():
                policy._zone_integral[zone]=200./scale
            initial=dict(policy._zone_integral)
            policy.get_action(observation([296.5]*4))
            self.assertEqual(policy._zone_integral,initial)
            policy.get_action(observation([295.4]*4))
            for zone in initial:self.assertLess(policy._zone_integral[zone],initial[zone])

    def test_control_reproduces_v6_and_each_standalone_matches_active_split(self):
        for key,_,_ in VARIANTS:
            source,_=assemble_variant(key)
            scope={}
            exec(compile(source,"split_policy.py","exec"),scope)
            registered=resolve_policy(key)(realistic_metadata(),None)
            standalone=scope["FinalPolicy"](realistic_metadata(),None)
            control=resolve_policy("geo6_threequarter")(realistic_metadata(),None) if key=="geo7_ctrl" else None
            for hour,temp in [(12,[296.5]*4),(12,[294.5]*4),(12,[299]*4),
                              (23,[295.5]*4),(5,[292,299,296,295])]*40:
                obs=observation(temp,hour,comfort=hour==12)
                action=registered.get_action(obs)
                for other in (standalone,control):
                    if other is not None:
                        comparison=other.get_action(obs)
                        for field in action:np.testing.assert_array_equal(action[field],comparison[field])

    def test_missing_geometry_falls_back_to_existing_controller(self):
        meta=metadata()
        del meta["zone_map"]
        left=resolve_policy("geo7_soft_i")(meta,None)
        right=resolve_policy("geo6_threequarter")(meta,None)
        for temp in ([292]*4,[299]*4,[295.5]*4)*40:
            a,b=left.get_action(observation(temp)),right.get_action(observation(temp))
            for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_invalid_exponents_and_disabled_integral_scaling_are_rejected(self):
        for exponent in (-1.,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                SplitGeometryPIPolicy(metadata(),None,integral_area_exponent=exponent)
        with self.assertRaises(ValueError):
            SplitGeometryPIPolicy(metadata(),None,scale_integral=False)


if __name__=="__main__":unittest.main()
