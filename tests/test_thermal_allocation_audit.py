import unittest
import numpy as np
from hvaclab.thermal_allocation_audit import cooling_candidate


class ThermalAllocationAuditTest(unittest.TestCase):
    def test_static_zone_cooling_is_preserved_with_lower_airflow(self):
        t=np.array([298.,297.,299.]);q=np.array([.4,.5,.3]);maxq=np.ones(3)*2;design=np.ones(3)*2000
        r=cooling_candidate(t,q,maxq,design,291.,289.)
        self.assertIsNotNone(r)
        np.testing.assert_allclose(np.array(r['flows'])*(t-289.),q*(t-291.),rtol=1e-12)
        self.assertTrue(np.all(np.array(r['flows'])<q))
        self.assertGreaterEqual(r['pressure'],200.)
        self.assertGreaterEqual(min(r['openings']),.1)
        self.assertLessEqual(max(r['openings']),1.)

    def test_infeasible_ventilation_floor_and_wrong_cooling_direction_are_rejected(self):
        self.assertIsNone(cooling_candidate([298.],[.0632455532],[2.],[2000.],291.,289.))
        self.assertIsNone(cooling_candidate([290.],[.5],[2.],[2000.],291.,289.))
        self.assertIsNone(cooling_candidate([298.],[.5],[2.],[2000.],291.,292.))
        self.assertIsNone(cooling_candidate([float('nan')],[.5],[2.],[2000.],291.,289.))


if __name__=='__main__':unittest.main()
