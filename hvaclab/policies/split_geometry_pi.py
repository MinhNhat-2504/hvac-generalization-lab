"""Separate geometry sensitivities for proportional and integral zone gains."""
import numpy as np

from hvaclab.policies.geometry_pi import GeometryPIPolicy, connected_room_areas


class SplitGeometryPIPolicy(GeometryPIPolicy):
    def __init__(self, building_meta_data, action_spec, integral_area_exponent=0.75,
                 reference_area_cv=2484., minimum_gain_scale=0.5,
                 maximum_gain_scale=2., scale_integral=True, **kwargs):
        if not np.isfinite(integral_area_exponent) or integral_area_exponent < 0:
            raise ValueError("integral area exponent must be finite and nonnegative")
        if not scale_integral:
            raise ValueError("split geometry requires scaled integral gains")
        super().__init__(building_meta_data, action_spec, reference_area_cv=reference_area_cv,
                         minimum_gain_scale=minimum_gain_scale,
                         maximum_gain_scale=maximum_gain_scale,
                         scale_integral=True, **kwargs)
        self._zone_integral_gain_scales = {}
        if self.geometry_status == "mapped":
            areas=connected_room_areas(building_meta_data["zone_map"])
            self._zone_integral_gain_scales = {
                f"VAV_{i}":float(np.clip((area/reference_area_cv)**integral_area_exponent,
                                          minimum_gain_scale,maximum_gain_scale))
                for i,area in enumerate(areas,1)
            }

    def _command_zones(self, continuous, zones, obs, low, high):
        if not (self.zone_pi_enabled and self._pi_active and self._zone_integral_gain_scales):
            return super()._command_zones(continuous,zones,obs,low,high)
        base_ki,base_limit=self.zone_ki,self.zone_integral_limit
        try:
            for zone in zones:
                p_scale=self._zone_gain_scales[zone[0]]
                i_scale=self._zone_integral_gain_scales[zone[0]]
                # The unchanged GeometryPIPolicy multiplies Ki by p_scale and
                # divides the raw integral limit by it. Precompensate that
                # ratio to obtain Ki*i_scale and limit/i_scale; its P gain,
                # conditional integration and actuator mapping are reused.
                ratio=i_scale/p_scale
                self.zone_ki=base_ki*ratio
                self.zone_integral_limit=base_limit/ratio
                super()._command_zones(continuous,[zone],obs,low,high)
        finally:
            self.zone_ki,self.zone_integral_limit=base_ki,base_limit
