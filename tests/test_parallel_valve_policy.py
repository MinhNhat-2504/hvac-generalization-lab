import unittest
import numpy as np
from hvaclab.parallel_valve_entry import SPECS,register,assemble
from hvaclab.registry import resolve_policy
from test_heat_delivery import building,observation


class ParallelValveTest(unittest.TestCase):
    def test_identity_and_each_standalone_match(self):
        register()
        for key in SPECS['parallel_valve']['arms']:
            source,params=assemble(key)
            scope={}
            exec(compile(source,key,'exec'),scope)
            left=resolve_policy(key)(building(),None)
            right=scope['FinalPolicy'](building(),None)
            baseline=resolve_policy('damper11_less')(building(),None) if key=='parallelv_ctrl' else None
            self.assertEqual(params['hold_steps'],432)
            self.assertEqual(params['heating_damper_exponent'],1.25)
            for hour,temp in [(12,[292,296,295,299]),(12,[299]*4),(23,[293]*4)]*80:
                obs=observation(temp,hour,comfort=hour==12)
                action=left.get_action(obs)
                for other in (right,baseline):
                    if other is not None:
                        comparison=other.get_action(obs)
                        for field in action:np.testing.assert_array_equal(action[field],comparison[field])


if __name__=='__main__':unittest.main()
