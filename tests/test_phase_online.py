import copy
from dataclasses import asdict
import unittest
import numpy as np
from hvaclab.policies.phase_online_actuator import PhaseOnlineActuatorPolicy,PHASES
from hvaclab.policies.heating_damper_curve import HeatingDamperCurvePolicy
from hvaclab.policies.online_actuator import OnlineActuatorPolicy
from hvaclab.phase_online_worker import verify_record
from hvaclab.runner import ScenarioResult
from hvaclab.pipeline_entry import register
from hvaclab.registry import _VARIANTS
from test_online_actuator import metadata,obs
from test_heat_delivery import observation


class PhaseOnlineTest(unittest.TestCase):
    def make(self,phase):
        register();return PhaseOnlineActuatorPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],online_blend=.25,online_warmup=0,feedback_phase=phase)

    def public_obs(self,hour,comfort,occupied):
        x=np.r_[observation([295.5]*4,hour,comfort=comfort),298.,1000.,.7,270.]
        x[5]=occupied;return x

    def test_phase_definitions_include_scheduled_empty(self):
        p=self.make('all')
        for hour,comfort,occupied,expected in ((5,False,0,'prestart'),(12,True,1,'occupied'),(12,True,0,'scheduled_empty'),(23,False,0,'offhours')):
            self.assertEqual(p.decision_phase(self.public_obs(hour,comfort,occupied)),expected)

    def test_all_reproduces_original_online_policy_during_learning(self):
        p=self.make('all');q=OnlineActuatorPolicy(metadata(),None,**_VARIANTS['damper12_mid'][1],online_blend=.25,online_warmup=0)
        for t in range(180):
            x=obs(t);a,b=p.get_action(x),q.get_action(x)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(p.online_report()['actual_correction_zone_steps'],q.correction_zone_steps)
        self.assertGreater(p.estimator.counts.sum(),0)

    def test_masks_change_only_allowed_phase_and_count_actual_application(self):
        for mode in ('prestart','occupied'):
            for hour,comfort,occupied in ((5,False,0),(12,True,1),(12,True,0)):
                p=self.make(mode);p._pi_active=True;p._correction[:]=.2
                x=self.public_obs(hour,comfort,occupied);phase=p.decision_phase(x)
                a=np.zeros(p.n_continuous,dtype=np.float32)
                b=a.copy();zones=p.ahu_controllers[0]['zones']
                reference=self.make(mode);reference._pi_active=True
                HeatingDamperCurvePolicy._command_zones(reference,b,zones,x,*p._comfort_band)
                p._command_zones(a,zones,x,*p._comfort_band)
                if phase==mode:
                    self.assertFalse(np.array_equal(a,b));self.assertEqual(p.online_report()['actual_correction_zone_steps'],4)
                else:
                    np.testing.assert_array_equal(a,b);self.assertEqual(p.online_report()['actual_correction_zone_steps'],0)
        p.reset();self.assertTrue(all(n==0 for n in p.applied_by_phase.values()))

    def test_learning_continues_while_feedback_phase_is_blocked(self):
        p=self.make('prestart')
        for t in range(180):p.get_action(obs(t))
        self.assertGreater(p.estimator.counts.sum(),0)
        self.assertGreater(p.online_report()['proposed_correction_zone_steps'],0)
        self.assertEqual(p.online_report()['actual_correction_zone_steps'],0)

    def test_phase_cost_validation_rejects_nonconservation(self):
        task=dict(building='b',weather='w',policy='phase_prestart',steps=4)
        row=asdict(ScenarioResult('b','w','phase_prestart',4,-.04,-.01,prod_regret=-.02,energy_cost=.08,carbon=.06))
        buckets={p:dict(steps=0,reward_sum=0.,prod_regret=0.,energy_cost=0.,carbon=0.) for p in PHASES}
        for p in ('prestart','occupied'):buckets[p].update(steps=2,reward_sum=-.02,prod_regret=-.01,energy_cost=.04,carbon=.03)
        record=dict(task=task,fingerprint='test',result=row,phase_sums=buckets,shadow_action_parity_steps=0)
        job=dict(task=task,fingerprint='test')
        verify_record(record,job)
        broken=copy.deepcopy(record);broken['phase_sums']['prestart']['reward_sum']=-.03
        with self.assertRaises(ValueError):verify_record(broken,job)


if __name__=='__main__':unittest.main()
