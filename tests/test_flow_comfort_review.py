import unittest
from hvaclab.flow_comfort_review import weighted_discomfort,compare_traces


class FlowComfortAttributionTest(unittest.TestCase):
    def test_piecewise_reward_uses_occupancy_and_uncapped_large_excursions(self):
        parts=weighted_discomfort({'hot':dict(temperature=307.,low=294.,high=297.,occupancy=1.),
                                   'cold':dict(temperature=293.,low=294.,high=297.,occupancy=3.),
                                   'empty':dict(temperature=200.,low=294.,high=297.,occupancy=0.)})
        self.assertAlmostEqual(parts['hot'],.6*.25*2)
        self.assertAlmostEqual(parts['cold'],.6*.75*.2)

    def test_empty_building_has_zero_comfort_and_unmatched_trace_is_rejected(self):
        self.assertEqual(weighted_discomfort({'z':dict(temperature=300.,low=294.,high=297.,occupancy=0.)}),dict(hot=0.,cold=0.))
        with self.assertRaises(ValueError):compare_traces(dict(schema={},rows=[]),dict(schema={},rows=[]),True)

    def test_paired_trace_attribution_conserves_delta_and_finds_first_difference(self):
        def row(i):return dict(epoch=i*300,observed_temperature=[298.],run=[1],hold_left=[432],sat=[-.5],
            phase='occupied',comfort=.12,zones={'r':dict(temperature=298.,low=294.,high=297.,occupancy=1.)})
        a=[row(i) for i in range(8640)];b=list(a)
        b[100]={**a[100],'observed_temperature':[299.],'comfort':.24,
                'zones':{'r':dict(temperature=299.,low=294.,high=297.,occupancy=1.)}}
        r=compare_traces(dict(schema={},rows=a),dict(schema={},rows=b),True)
        self.assertAlmostEqual(r['comfort_delta'],.12/8640)
        self.assertEqual(r['first_events']['observed_temperature_0.1K']['step'],100)
        self.assertNotIn('ahu_run_mismatch',r['first_events'])
        self.assertAlmostEqual(r['zone_deltas'][0][1],r['comfort_delta'])
        b[100]['zones']['r']['occupancy']=2.
        with self.assertRaises(ValueError):compare_traces(dict(schema={},rows=a),dict(schema={},rows=b),True)


if __name__=='__main__':unittest.main()
