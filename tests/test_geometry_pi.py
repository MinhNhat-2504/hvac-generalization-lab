import unittest

import numpy as np

from hvaclab.policies.geometry_pi import GeometryPIPolicy, connected_room_areas
from hvaclab.registry import resolve_policy
from hvaclab.submission.experimental import assemble_variant
from test_heat_delivery import building, observation


def metadata():
    meta=building()
    meta["observation_field_names"]=[
        f"VAV_{int(n.split('_')[1])+1}_zone_air_temperature_sensor" if n.startswith("VAV_") else n
        for n in meta["observation_field_names"]]
    meta["zone_action_indices"]={f"VAV_{i+1}":v for i,v in enumerate(meta["zone_action_indices"].values())}
    meta["ahu_zone_assignments"]={"1":[f"room_{i}" for i in range(1,5)]}
    grid=np.full((10,12),2.)
    grid[1,1]=0
    grid[1:3,4:6]=0
    grid[5:8,1:4]=0
    grid[5:9,6:10]=0
    meta["zone_map"]=grid
    return meta


class GeometryPITest(unittest.TestCase):
    def test_component_areas_are_four_connected_and_do_not_mutate_map(self):
        grid=metadata()["zone_map"]
        saved=grid.copy()
        self.assertEqual(connected_room_areas(grid),[1,4,9,16])
        np.testing.assert_array_equal(grid,saved)
        self.assertEqual(connected_room_areas(np.array([[0,2],[2,0]])),[1,1])

    def test_zone_gain_scaling_and_base_attributes_restore(self):
        for scale_i in (False,True):
            policy=GeometryPIPolicy(metadata(),None,reference_area_cv=4.,scale_integral=scale_i)
            self.assertEqual(policy.geometry_status,"mapped")
            policy.get_action(observation([296.5]*4))
            for i,scale in enumerate((.5,1.,1.5,2.),1):
                expected=.1*scale+.005*(scale if scale_i else 1.)
                self.assertAlmostEqual(policy._zone_u[f"VAV_{i}"],expected)
            self.assertEqual((policy.zone_kp,policy.zone_ki,policy.zone_integral_limit),(.1,.005,200.))

    def test_scaled_integral_retains_full_authority(self):
        policy=GeometryPIPolicy(metadata(),None,reference_area_cv=4.,scale_integral=True)
        for vav,scale in policy._zone_gain_scales.items():
            policy._zone_integral[vav]=200./scale
        policy.get_action(observation([295.5]*4))
        for u in policy._zone_u.values():
            self.assertAlmostEqual(u,1.)

    def test_unknown_mapping_falls_back_to_v4(self):
        meta=metadata()
        meta["zone_map"]=np.zeros((2,2))
        candidate=resolve_policy("geo_p_sqrt")(meta,None)
        baseline=resolve_policy("e_sync")(meta,None)
        self.assertEqual(candidate.geometry_status,"mapping_mismatch")
        for temp in ([292]*4,[299]*4,[295.5]*4)*30:
            a,b=baseline.get_action(observation(temp)),candidate.get_action(observation(temp))
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])

    def test_v6_controls_and_standalone_use_declared_geometry_exponents(self):
        meta=metadata()
        zone_map=np.ones((102,260),dtype=int)
        column=1
        for side in (25,50,75,100):
            zone_map[1:side+1,column:column+side]=0
            column+=side+1
        meta["zone_map"]=zone_map
        for key,exponent in (("geo6_ctrl",.5),("geo6_quarter",.25),
                             ("geo6_threequarter",.75),("geo6_linear",1.)):
            policy=resolve_policy(key)(meta,None)
            expected=np.clip((np.array([625,2500,5625,10000])/2484.)**exponent,.5,2.)
            np.testing.assert_allclose(list(policy._zone_gain_scales.values()),expected)
            source,_=assemble_variant(key)
            scope={}
            exec(compile(source,"v6_policy.py","exec"),scope)
            standalone=scope["FinalPolicy"](meta,None)
            reference=resolve_policy("geo_pi_sqrt")(meta,None) if key=="geo6_ctrl" else None
            for temp in ([296.5]*4,[294.5]*4,[299]*4)*30:
                obs=observation(temp)
                action=policy.get_action(obs)
                for other in (standalone,reference):
                    if other is not None:
                        comparison=other.get_action(obs)
                        for field in action:
                            np.testing.assert_array_equal(action[field],comparison[field])

    def test_standalone_matches_active_geometry_scaling(self):
        for key in ("geo_ctrl","geo_p_sqrt","geo_pi_sqrt","geo_p_linear"):
            source,_=assemble_variant(key)
            scope={}
            exec(compile(source,"geometry_candidate.py","exec"),scope)
            # Use the synthetic area scale so all clipped gain branches occur.
            left=scope["GeometryPIPolicy"](metadata(),None,reference_area_cv=4.,scale_integral=key=="geo_pi_sqrt")
            right=GeometryPIPolicy(metadata(),None,reference_area_cv=4.,scale_integral=key=="geo_pi_sqrt")
            for temp in ([296.5]*4,[294.5]*4,[299]*4)*10:
                a,b=left.get_action(observation(temp)),right.get_action(observation(temp))
                for field in a:
                    np.testing.assert_array_equal(a[field],b[field])
            assembled=scope["FinalPolicy"](metadata(),None)
            registered=resolve_policy(key)(metadata(),None)
            for temp in ([296.5]*4,[294.5]*4,[299]*4):
                a,b=assembled.get_action(observation(temp)),registered.get_action(observation(temp))
                for field in a:
                    np.testing.assert_array_equal(a[field],b[field])


if __name__ == "__main__":
    unittest.main()
