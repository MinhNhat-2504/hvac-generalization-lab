import copy
import unittest
import numpy as np
from hvaclab.policies.thermal_response_model import StateThermalPredictor,fan_state
from hvaclab.thermal_response_study import examples


class ThermalResponseTest(unittest.TestCase):
    def trace(self):
        return dict(zone_ids=['zone'],ahu_zone_indices=[[0]],trace=[dict(
            observed_before=[float(t*t)],ahu_on=[t%2],occupancy=1.,physical_after=[999.]) for t in range(8)])

    def test_delayed_increment_uses_previous_command(self):
        rows=list(examples(self.trace(),5))
        self.assertEqual(rows[0],(0,'on_entry',4.,3.,1.,5.))
        self.assertEqual(rows[1][1],'off_entry')
        self.assertEqual(len(rows),2)

    def test_fitting_examples_ignore_private_field_and_future_suffix(self):
        a=self.trace();b=copy.deepcopy(a)
        for row in b['trace']:row['physical_after']=[-999.]
        for row in b['trace'][5:]:row['observed_before']=[-999.]
        self.assertEqual(list(examples(a,5)),list(examples(b,5)))

    def test_fit_separates_on_off_response_and_pools_sparse_zone(self):
        samples=[]
        for _ in range(200):
            samples.extend([(0,'off_entry',295.5,0.,1.,.2),(0,'on_entry',295.5,0.,1.,-.1)])
        model=StateThermalPredictor().fit(samples)
        self.assertAlmostEqual(model.predict_delta(0,'off_entry',295.5,0,1),.2,places=3)
        self.assertAlmostEqual(model.predict_delta(0,'on_entry',295.5,0,1),-.1,places=3)
        self.assertGreater(model.predict_delta(99,'off_entry',295.5,0,1),0.)
        self.assertIsNone(model.predict_delta(0,'off_steady',295.5,0,1))
        self.assertIsNone(model.predict_delta(0,'off_entry',np.nan,0,1))

    def test_hybrid_uses_state_model_only_at_command_transition(self):
        from hvaclab.thermal_response_hybrid import hybrid
        rows=[dict(state=state,state_model=1.,linear=2.) for state in ('on_entry','off_entry','on_steady','off_steady')]
        result=hybrid(rows)
        self.assertEqual([r['state_model'] for r in result],[1.,1.,2.,2.])
        self.assertTrue(all(r['state_model']==1. for r in rows))

    def test_forecaster_refuses_missing_or_too_wide_intervals(self):
        from hvaclab.policies.thermal_response_forecast import ThermalResponseForecaster
        class ZeroModel:
            def predict_delta(self,*args):return 0.
        args=dict(zone=0,observed=295.5,previous_observed=295.5,occupied=1.,
                  previous_on=True,before_previous_on=True,next_on=False)
        missing=ThermalResponseForecaster(ZeroModel(),{}).predict(**args)
        self.assertIsNone(missing['interval'])
        self.assertFalse(missing['interval_allows_coast'])
        wide=ThermalResponseForecaster(ZeroModel(),{'off_entry':1.84}).predict(**args)
        self.assertFalse(wide['interval_allows_coast'])
        narrow=ThermalResponseForecaster(ZeroModel(),{'off_entry':.1}).predict(**args)
        self.assertTrue(narrow['interval_allows_coast'])
        class OutsideCurrent:
            def __init__(self):self.calls=0
            def predict_delta(self,*args):
                self.calls+=1
                return 2. if self.calls==1 else -2.
        unsafe_current=ThermalResponseForecaster(OutsideCurrent(),{'off_entry':.1}).predict(**args)
        self.assertFalse(unsafe_current['interval_allows_coast'])


    def test_forecaster_matches_evaluated_two_step_model(self):
        from hvaclab.policies.thermal_response_forecast import ThermalResponseForecaster
        from hvaclab.thermal_response_study import forecasts
        run=self.trace();run['comfort_band']=[294.,297.]
        model=StateThermalPredictor().fit(examples(run,5))
        forecaster=ThermalResponseForecaster(model,{},transition_only=False)
        for row in forecasts(run,model,2,8):
            t=row['step'];trace=run['trace']
            result=forecaster.predict(0,trace[t]['observed_before'][0],trace[t-1]['observed_before'][0],1.,
                trace[t-1]['ahu_on'][0],trace[t-2]['ahu_on'][0],trace[t]['ahu_on'][0])
            self.assertAlmostEqual(result['next_temperature'],row['state_model'])

    def test_entry_and_steady_states(self):
        self.assertEqual(fan_state(False,True),'off_entry')
        self.assertEqual(fan_state(True,False),'on_entry')
        self.assertEqual(fan_state(False,False),'off_steady')
        self.assertEqual(fan_state(True,True),'on_steady')


if __name__=='__main__':unittest.main()
