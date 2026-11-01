import unittest
from unittest.mock import patch
import numpy as np
from hvaclab.registry import resolve_policy
from hvaclab.policies.zone_pi import ZonePIPolicy
from hvaclab.submission.experimental import assemble_variant
from test_heat_delivery import building,observation


class PersistenceTest(unittest.TestCase):
    def test_control_and_standalones_match(self):
        for key in ('persist_ctrl','persist_144','persist_432','persist_576'):
            source,_=assemble_variant(key)
            scope={}
            exec(compile(source,key,'exec'),scope)
            left=resolve_policy(key)(building(),None)
            right=scope['FinalPolicy'](building(),None)
            baseline=resolve_policy('valve8_gentle')(building(),None) if key=='persist_ctrl' else None
            for temps in ([299]*4,[295.5]*4,[292]*4)*100:
                obs=observation(temps)
                a=left.get_action(obs)
                for other in (right,baseline):
                    if other is not None:
                        b=other.get_action(obs)
                        for field in a:np.testing.assert_array_equal(a[field],b[field])

    def test_hold_duration_and_cooling_refresh_are_exact(self):
        for key,steps in (('persist_144',144),('persist_432',432),('persist_576',576)):
            policy=resolve_policy(key)(building(),None)
            def action(running):
                cont=-np.ones(13,dtype=np.float32)
                disc=np.array([int(running),0])
                cont[0]=-.5 if running else -1.
                return dict(continuous_action=cont,discrete_action=disc)
            obs=observation([295.5]*4,hour=23,comfort=False)
            with patch.object(ZonePIPolicy,'get_action',side_effect=lambda obs:action(True)):
                policy.get_action(obs)
                self.assertEqual(policy._hold_left,[steps])
            with patch.object(ZonePIPolicy,'get_action',side_effect=lambda obs:action(False)):
                for _ in range(steps):self.assertEqual(policy.get_action(obs)['discrete_action'][0],1)
                self.assertEqual(policy.get_action(obs)['discrete_action'][0],0)


if __name__=='__main__':unittest.main()
