import unittest
from hvaclab.paper_validation import tasks,ARMS,WEATHER


class PaperValidationTest(unittest.TestCase):
    def test_full_cross_product_without_duplicate_or_short_episode(self):
        jobs=tasks([f'building_{i}' for i in range(12)])
        self.assertEqual(len(jobs),192)
        self.assertEqual(len({(x['building'],x['weather'],x['policy']) for x in jobs}),192)
        self.assertTrue(all(x['steps']==8640 for x in jobs))
        self.assertEqual(len(ARMS),4);self.assertEqual(len(WEATHER),4)

    def test_flow_control_disables_thermal_intervention(self):
        from hvaclab.pipeline_entry import register
        from hvaclab.guarded_lift_worker import register as guarded
        from hvaclab.paper_validation_worker import register as extra
        from hvaclab.registry import _VARIANTS
        register();guarded();extra();params=_VARIANTS['flow_only'][1]
        self.assertTrue(params['flow_equivalent'])
        self.assertFalse(params['cooling_lift']);self.assertFalse(params['project_lift'])

    def test_cluster_uncertainty_uses_buildings_not_steps(self):
        from hvaclab.paper_validation_review import cluster_interval
        result=cluster_interval([1.,2.,3.],[.9,1.8,2.7],draws=100)
        self.assertEqual(result['clusters'],3)
        self.assertAlmostEqual(result['mean_change_pct'],-10.)
        for endpoint in result['cluster_95pct_interval']:self.assertAlmostEqual(endpoint,-10.)
        with self.assertRaises(ValueError):cluster_interval([1.],[.9])
