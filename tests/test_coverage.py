import json
from contextlib import contextmanager
import uuid
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from hvaclab import coverage
from hvaclab.runner import ScenarioResult


@contextmanager
def workspace_directory():
    root = Path(__file__).resolve().parent / ("coverage_test_" + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        # Only remove files generated inside this uniquely created directory.
        for path in root.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            path.rmdir()
        root.rmdir()


class CoverageTest(unittest.TestCase):
    def test_short_failed_and_nonfinite_results_are_not_resumable(self):
        task = dict(building="b", weather="w", policy="p", steps=8640)
        row = asdict(ScenarioResult("b", "w", "p", 8640, -8640., -1.))
        self.assertTrue(coverage.valid_result(row, task))
        for changes in ({"steps": 12}, {"error": "failure"},
                        {"mean_reward": float("nan")}, {"total_reward": -1.}):
            self.assertFalse(coverage.valid_result({**row, **changes}, task))

    def test_completed_job_resumes_without_starting_an_interpreter(self):
        self.exercise_resume(stale=False)

    def test_changed_result_provenance_is_rejected(self):
        self.exercise_resume(stale=True)

    def exercise_resume(self, stale):
        with workspace_directory() as directory:
            root = Path(directory)
            cache = root / "cache.json"
            cache.write_text("{}")
            plan = root / "plan.json"
            plan.write_text(json.dumps({"splits": {"training": {"cache_file": str(cache)}}}))
            task = dict(split="training", building="b", weather="w", policy="p", steps=8640,
                        scenarios_dir="unused")
            row = asdict(ScenarioResult("b", "w", "p", 8640, -8640., -1.))
            output = root / "out"

            class Worker:
                pid = 123

                def __init__(self, command, **kwargs):
                    job = json.loads(Path(command[-1]).read_text())
                    coverage.write_json(job["result_path"], dict(task=job["task"],
                                        fingerprint=job["fingerprint"], result=row))

                def poll(self):
                    return 0

            with patch.object(coverage, "prepare", return_value=(
                    [task], {"training": []}, {"training": {("b", "w")}})), \
                    patch.object(coverage, "source_snapshot", return_value={}), \
                    patch.object(coverage.subprocess, "Popen", side_effect=Worker) as launch:
                coverage.run(plan, output, 1)
                self.assertEqual(launch.call_count, 1)
                job_path = next((output / "jobs").glob("*.result.json"))
                if stale:
                    record = json.loads(job_path.read_text())
                    record["fingerprint"] = "wrong-source"
                    coverage.write_json(job_path, record)
                    with self.assertRaisesRegex(ValueError, "stale result"):
                        coverage.run(plan, output, 1)
                else:
                    coverage.run(plan, output, 1)
                    self.assertEqual(launch.call_count, 1)
                    status = json.loads((output / "status.json").read_text())
                    self.assertEqual(status["status"], "complete")
                    self.assertEqual(status["counts"]["training"]["completed"], 1)


if __name__ == "__main__":
    unittest.main()
