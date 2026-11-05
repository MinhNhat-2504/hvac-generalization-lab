import unittest
import numpy as np
from hvaclab.flow_equivalent_full_review import cohort_statistics,full_mean_gate
from hvaclab.flow_equivalent_package import assemble
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy
from test_online_actuator import metadata
from test_heat_delivery import observation


def row(building='building_train_1',weather='a',loss=.02,comfort=.005):
    return dict(building=building,weather=weather,steps=8640,mean_reward=-loss,
                prod_regret=-comfort*8640/.6,energy_cost=(loss-comfort)*8640/.4,carbon=(loss-comfort)*8640/.4,error='')


class FullFlowGateTest(unittest.TestCase):
    def test_lower_total_with_worse_comfort_does_not_pass(self):
        a=row();b=row(loss=.019,comfort=.006)
        self.assertFalse(cohort_statistics([b],[a])['passes_mean_gate'])

    def test_missing_duplicate_and_short_episodes_are_rejected(self):
        with self.assertRaises(ValueError):cohort_statistics([],[row()])
        with self.assertRaises(ValueError):cohort_statistics([row(),row()],[row()])
        with self.assertRaises(ValueError):cohort_statistics([{**row(),'steps':2016}],[row()])
        with self.assertRaises(ValueError):full_mean_gate([row(loss=.019)],[row()])

    def test_both_full_groups_must_improve_and_partial_cohorts_never_pass(self):
        base=[row('building_train_'+str(i),loss=.02) for i in range(30)]+[row('building_'+str(i),loss=.02) for i in range(20)]
        candidate=[{**r,'mean_reward':-.019} for r in base]
        self.assertTrue(full_mean_gate(candidate,base))
        candidate[-1]['mean_reward']=-.05
        self.assertFalse(full_mean_gate(candidate,base))

    def test_standalone_matches_local_policy_through_active_transform(self):
        source,params=assemble();namespace={};exec(compile(source,'flow_standalone','exec'),namespace)
        m=metadata();m['action_normalization_metadata']['1_supply_air_static_pressure_setpoint']=dict(min_native_value=0.,max_native_value=2000.,min_normalized_value=-1.,max_normalized_value=1.)
        p=namespace['FinalPolicy'](m,None);q=FlowEquivalentPolicy(m,None,**params)
        for temp in ([299.]*4,[295.5]*4,[292.]*4,[296.]*4)*25:
            x=np.r_[observation(temp),298.,1000.,.7,270.]
            a,b=p.get_action(x),q.get_action(x)
            for field in a:np.testing.assert_array_equal(a[field],b[field])
        self.assertGreater(p.rebalanced_handler_steps,0)


if __name__=='__main__':unittest.main()
