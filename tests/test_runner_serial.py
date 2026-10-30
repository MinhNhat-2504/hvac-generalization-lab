import unittest
from dataclasses import asdict
from unittest.mock import patch

from hvaclab import runner


class SerialRunnerTest(unittest.TestCase):
    def test_serial_path_runs_each_task_without_creating_a_pool(self):
        def result(task):
            building, weather, policy, steps, _ = task
            return asdict(runner.ScenarioResult(
                building, weather, policy, steps, -float(steps), -1.0))

        with patch.object(runner, "_run_one", side_effect=result) as run_one:
            with patch.object(runner.mp, "get_context") as context:
                rows = runner.run_sweep(
                    ["a", "b"], [("building", "weather")],
                    num_steps=12, workers=1, progress=False,
                )
        context.assert_not_called()
        self.assertEqual(run_one.call_count, 2)
        self.assertEqual([r.policy for r in rows], ["a", "b"])
        self.assertEqual([r.steps for r in rows], [12, 12])
        self.assertEqual(runner.summarise(rows)["a"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
