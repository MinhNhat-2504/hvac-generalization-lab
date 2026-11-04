import copy
import unittest
import numpy as np
from hvaclab.actuator_response_study import PublicActuatorView,records,masked
from hvaclab.policies.actuator_thermal_model import actuator_features,ActuatorThermalPredictor,FEATURE_NAMES


def payload():
    names=['num_occupants','zone_temp','ahs_ahu_1_supply_air_temperature_sensor',
        'ahs_ahu_1_supply_air_static_pressure_sensor','ahs_ahu_1_recirculation_percentage_command',
        'ahs_ahu_1_outside_air_temperature_sensor']
    schema=dict(field_names=names,observation_normalization_metadata={'zone_air_temperature_sensor':{'sample_mean':295.,'sample_std':5.}},
        action_normalization_metadata={'supply_water_temperature_setpoint':dict(min_native_value=300.,max_native_value=360.,min_normalized_value=-1.,max_normalized_value=1.)},
        zones=[dict(index=1,handler=0,actions={'supply_air_damper_percentage_command':3,'reheat_valve_percentage_command':4})],
        controllers=[dict(ahu_id='ahu_1',run_discrete_idx=0,temp_idx=0,pressure_idx=1,recirc_idx=2)],
        water_discrete_idx=1,water_temp_idx=5,water_pressure_idx=6)
    rows=[dict(observation=[1.,t*.02,298.,1000.,.7,270.],continuous=[t*.1,-1.,1.,0.,0.,0.,-1.],discrete=[t%2,1]) for t in range(8)]
    return dict(schema=schema,rows=rows)


class ActuatorThermalTest(unittest.TestCase):
    def test_public_context_uses_previous_command_and_raw_sat_proxy(self):
        view=PublicActuatorView(payload());context=view.context(2,0)
        self.assertAlmostEqual(context['sat_command'],.1)
        self.assertEqual(context['sat_proxy_k'],298.)
        self.assertEqual(context['outside_k'],270.)
        self.assertEqual(context['water_temperature_native'],330.)
        self.assertEqual(context['damper'],.5)
        self.assertEqual(view.state(2,0),'on_entry')

    def test_no_current_command_or_future_suffix_leaks_into_features(self):
        a=payload();b=copy.deepcopy(a)
        b['rows'][2]['continuous']=[-999.]*7
        x=list(records(PublicActuatorView(a),2,4))[0]
        y=list(records(PublicActuatorView(b),2,4))[0]
        np.testing.assert_array_equal(x['features'],y['features'])
        self.assertAlmostEqual(x['target']-x['temperature'],.1)
        b=copy.deepcopy(a)
        for row in b['rows'][5:]:row['observation']=[-999.]*6
        ar=list(records(PublicActuatorView(a),2,5));br=list(records(PublicActuatorView(b),2,5))
        for u,v in zip(ar,br):np.testing.assert_array_equal(u['features'],v['features']);self.assertEqual(u['target'],v['target'])

    def test_features_and_ablations_have_fixed_meaning(self):
        view=PublicActuatorView(payload());x=actuator_features(295.,.2,1.,view.context(2,0))
        self.assertEqual(x.shape,(len(FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(x)))
        base=masked(x,'state_only');commands=masked(x,'commands_only')
        np.testing.assert_array_equal(base[:4],x[:4]);self.assertTrue(np.all(base[4:]==0))
        self.assertTrue(np.all(commands[13:]==0))
        off=view.context(2,0);off['fan']=0
        values=dict(zip(FEATURE_NAMES,actuator_features(295.,.2,1.,off)))
        self.assertEqual(values['mixed_cooling_proxy'],0.)
        self.assertEqual(values['reheat_lift_proxy'],0.)

    def test_prefixed_normalization_mismatch_is_rejected(self):
        data=payload();data['rows'][2]['observation'][2]=.6
        with self.assertRaises(ValueError):PublicActuatorView(data).context(2,0)

    def test_adaptive_selection_uses_calibration_regimes_and_five_percent_gate(self):
        from hvaclab.policies.adaptive_thermal_forecast import AdaptiveThermalSelector
        occupied=dict(state='on_steady',occupancy=1.,target=295.,aged_sensor=295.1,
                      linear=295.05,state_only=295.03,commands_only=295.04,full_actuators=295.01)
        empty=dict(state='on_steady',occupancy=0.,target=295.,aged_sensor=295.2,
                   linear=295.1,state_only=295.3,commands_only=295.097,full_actuators=295.099)
        selector=AdaptiveThermalSelector(minimum_samples=2).fit([occupied,occupied,empty,empty])
        self.assertEqual(selector.predict(occupied)['chosen'],'full_actuators')
        self.assertEqual(selector.predict(empty)['chosen'],'linear')
        unseen={**occupied,'state':'off_entry'}
        result=selector.predict(unseen)
        self.assertEqual(result['chosen'],'aged_sensor');self.assertIsNone(result['interval'])
        forecast=dict(occupied);forecast.pop('target')
        self.assertEqual(selector.predict(forecast)['temperature'],occupied['full_actuators'])

    def test_regression_and_missing_state_are_explicit(self):
        x=np.zeros(len(FEATURE_NAMES));x[0]=1.
        model=ActuatorThermalPredictor().fit((0,'on_steady',x,.2) for _ in range(1000))
        self.assertAlmostEqual(model.predict_delta(0,'on_steady',x),.2,places=3)
        self.assertIsNone(model.predict_delta(0,'off_entry',x))
        self.assertIsNone(model.predict_delta(0,'on_steady',np.zeros(1)))


if __name__=='__main__':unittest.main()
