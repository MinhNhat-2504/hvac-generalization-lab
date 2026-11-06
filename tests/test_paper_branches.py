"""Exact branch validation must expose small drift and reject invalid state."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'reports/scripts/probe_paper_branches.py'
spec=importlib.util.spec_from_file_location('branch_probe',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BranchDifferenceTest(unittest.TestCase):
    def test_single_ulp_drift_is_not_accepted(self):
        a=np.array([1.,2.]);b=a.copy();b[1]=np.nextafter(b[1],np.inf)
        report=module.array_difference(a,b)
        self.assertFalse(report['exact'])
        self.assertEqual(report['different_values'],1)
        self.assertGreater(report['max_absolute'],0)

    def test_identity_and_empty_state(self):
        for a in ([1.,2.],[]):
            self.assertEqual(module.array_difference(a,a),
                dict(exact=True,different_values=0,max_absolute=0.))

    def test_shape_and_nonfinite_rejected(self):
        for a,b in (([1.],[[1.]]),([np.nan],[np.nan]),([1.],[np.inf])):
            with self.assertRaises(ValueError):module.array_difference(a,b)

    def test_compiled_rng_replay_requires_its_own_state(self):
        from numba import njit
        @njit
        def draw():return np.random.random(8)
        draw()  # Compile before taking the state.
        original=module.numba_numpy_state()
        try:
            expected=draw();advanced=draw()
            self.assertFalse(np.array_equal(expected,advanced))
            module.numba_numpy_state(original)
            self.assertTrue(np.array_equal(expected,draw()))
        finally:module.numba_numpy_state(original)


if __name__=='__main__':unittest.main()
