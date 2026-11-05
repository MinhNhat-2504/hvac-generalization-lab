import unittest
from hvaclab.flow_equivalent_validation import fan_audit_passes


class FanMeterToleranceTest(unittest.TestCase):
    def test_float_roundoff_at_large_accumulated_meter_is_tolerated(self):
        self.assertTrue(fan_audit_passes(dict(parent_pressure_fan_Wh=447092.80681799015,
                                            actual_fan_Wh=447092.8071217131)))

    def test_actual_energy_increase_and_nonfinite_data_still_fail(self):
        self.assertFalse(fan_audit_passes(dict(parent_pressure_fan_Wh=100.,actual_fan_Wh=100.01)))
        self.assertFalse(fan_audit_passes(dict(parent_pressure_fan_Wh=100.,actual_fan_Wh=float('nan'))))
        self.assertFalse(fan_audit_passes(dict(parent_pressure_fan_Wh=-1.,actual_fan_Wh=0.)))


if __name__=='__main__':unittest.main()
