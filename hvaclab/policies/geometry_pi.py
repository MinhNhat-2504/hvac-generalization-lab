"""Geometry-scaled zone PI using only participant-facing static metadata.

Room IDs follow the starter kit's four-connected, row-major interior labels.
Unknown/missing mappings fall back to the unmodified zone PI. No simulator
object, reward signal, building name or weather label is read by this policy.
"""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


def connected_room_areas(zone_map):
    """Count four-connected interior components (0), in raster-root order.

    Uses numpy/stdlib only; the input map is never modified. The order and
    areas are checked offline against the starter kit's OpenCV component IDs.
    """
    array = np.asarray(zone_map)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        return []
    if not np.all(np.isfinite(array)) or not array.size:
        return []
    height, width = array.shape
    remaining = (array == 0).ravel()
    starts = np.flatnonzero(remaining)
    areas = []
    for value in starts:
        start = int(value)
        if not remaining[start]:
            continue
        remaining[start] = False
        stack = [start]
        count = 0
        while stack:
            index = stack.pop()
            count += 1
            col = index % width
            if index >= width and remaining[index-width]:
                remaining[index-width] = False
                stack.append(index-width)
            if index+width < height*width and remaining[index+width]:
                remaining[index+width] = False
                stack.append(index+width)
            if col and remaining[index-1]:
                remaining[index-1] = False
                stack.append(index-1)
            if col+1 < width and remaining[index+1]:
                remaining[index+1] = False
                stack.append(index+1)
        areas.append(count)
    return areas


class GeometryPIPolicy(CoordinatedHeatPolicy):
    def __init__(self, building_meta_data, action_spec, area_gain_enabled=True,
                 reference_area_cv=2484.0, area_exponent=0.5,
                 scale_integral=False, minimum_gain_scale=0.5,
                 maximum_gain_scale=2.0, **kwargs):
        if not np.isfinite(reference_area_cv) or reference_area_cv <= 0:
            raise ValueError("reference area must be positive")
        if not np.isfinite(area_exponent) or area_exponent < 0:
            raise ValueError("area exponent must be nonnegative")
        if not 0 < minimum_gain_scale <= maximum_gain_scale or not np.isfinite(maximum_gain_scale):
            raise ValueError("invalid gain-scale limits")
        super().__init__(building_meta_data, action_spec, **kwargs)
        self.area_gain_enabled = bool(area_gain_enabled)
        self.scale_integral = bool(scale_integral)
        self._zone_gain_scales = {}
        self.geometry_status = "disabled"
        if not self.area_gain_enabled:
            return
        zone_map = building_meta_data.get("zone_map")
        if zone_map is None:
            self.geometry_status = "missing_map"
            return
        areas = connected_room_areas(zone_map)
        observed = {vav for ctrl in self.ahu_controllers for vav, _, _ in ctrl["zones"]}
        expected = {f"VAV_{i}" for i in range(1, len(areas)+1)}
        if not observed or observed != expected:
            self.geometry_status = "mapping_mismatch"
            return
        self._zone_gain_scales = {
            f"VAV_{i}": float(np.clip((area/reference_area_cv)**area_exponent,
                                      minimum_gain_scale, maximum_gain_scale))
            for i, area in enumerate(areas, 1)
        }
        self.geometry_status = "mapped"

    def _command_zones(self, continuous, zones, obs, low, high):
        if not (self.area_gain_enabled and self.zone_pi_enabled and self._pi_active
                and self._zone_gain_scales):
            return super()._command_zones(continuous, zones, obs, low, high)
        base_kp, base_ki, base_limit = self.zone_kp, self.zone_ki, self.zone_integral_limit
        try:
            for zone in zones:
                scale = self._zone_gain_scales.get(zone[0], 1.)
                integral_scale = scale if self.scale_integral else 1.
                self.zone_kp = base_kp*scale
                self.zone_ki = base_ki*integral_scale
                self.zone_integral_limit = base_limit/integral_scale
                super()._command_zones(continuous, [zone], obs, low, high)
        finally:
            self.zone_kp, self.zone_ki, self.zone_integral_limit = base_kp, base_ki, base_limit
