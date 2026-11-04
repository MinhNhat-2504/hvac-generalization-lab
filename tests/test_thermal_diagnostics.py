import json
import unittest
from hvaclab.thermal_diagnostics import analyze_probe
from test_coverage import workspace_directory


class DiagnosticTest(unittest.TestCase):
    def test_sensor_alignment_and_restart_window(self):
        with workspace_directory() as root:
            rows=[]
            for step,(coast,after) in enumerate(((False,295.),(True,295.),(False,293.8))):
                rows.append(dict(step=step,observed_after=[295.],physical_before=[295.],
                                 physical_after=[after],coasting=[coast],ahu_on=[False]))
            path=root/'probe.json'
            path.write_text(json.dumps(dict(policy='probe',steps=3,comfort_band=[294.,297.],
                wall_seconds=10.,cpu_seconds=9.,policy_seconds=1.,environment_seconds=8.,
                ahu_zone_indices=[[0]],trace=rows)))
            result=analyze_probe(path)
            self.assertEqual(result['max_sensor_error_vs_pre_step'],0.)
            self.assertAlmostEqual(result['max_sensor_error_vs_post_step'],1.2)
            self.assertEqual(result['complete_coast_episodes'],1)
            self.assertEqual(result['guard_releases_without_ahu_restart'],1)
            self.assertEqual(result['episodes'][0]['duration_steps'],1)
            self.assertEqual(result['episodes_with_excursion_after_restart'],1)


if __name__=='__main__':unittest.main()
