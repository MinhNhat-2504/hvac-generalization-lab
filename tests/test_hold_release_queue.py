import unittest
import numpy as np
from hvaclab.hold_release_queue import ready
from hvaclab.hold_release_entry import register,assemble,ARMS
from hvaclab.registry import resolve_policy
from test_heat_delivery import building,observation


class QueueTest(unittest.TestCase):
    def test_dependency_must_complete_and_exit(self):
        for status in ('running','failed','interrupted','prepared'):
            self.assertFalse(ready({'status':status},False))
        self.assertFalse(ready({'status':'complete'},True))
        self.assertTrue(ready({'status':'complete'},False))

    def test_isolated_assembly_matches_all_registered_arms(self):
        register()
        for key in ARMS:
            source,_=assemble(key)
            scope={}
            exec(compile(source,key,'exec'),scope)
            left=scope['FinalPolicy'](building(),None)
            right=resolve_policy(key)(building(),None)
            for temps in ([299]*4,[293,295,295,295],[295.5]*4)*80:
                a,b=left.get_action(observation(temps)),right.get_action(observation(temps))
                for field in a:np.testing.assert_array_equal(a[field],b[field])


if __name__=='__main__':unittest.main()
