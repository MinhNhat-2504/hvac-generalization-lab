from contextlib import redirect_stdout
from dataclasses import asdict
import io
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import psutil

from hvaclab import pi_calibration as study
from hvaclab.coverage import write_json
from hvaclab.runner import ScenarioResult, save
from test_coverage import workspace_directory


class RunLockTest(unittest.TestCase):
    def test_live_owner_cannot_be_started_twice(self):
        with workspace_directory() as root:
            path=root/"scheduler.lock"
            with study.RunLock(path):
                with self.assertRaisesRegex(RuntimeError,"already running"):
                    with study.RunLock(path):
                        self.fail("duplicate scheduler admitted")
            self.assertFalse(path.exists())

    def test_stale_pid_reuse_is_recovered(self):
        with workspace_directory() as root:
            path=root/"scheduler.lock"
            owner=dict(pid=psutil.Process().pid,process_created_at=psutil.Process().create_time()-1000)
            write_json(path,owner)
            with study.RunLock(path):
                current=json.loads(path.read_text())
                self.assertNotEqual(current["process_created_at"],owner["process_created_at"])
            self.assertFalse(path.exists())


class CalibrationWorkflowTest(unittest.TestCase):
    def exercise(self,all_dev_fail,secondary_wins=False):
        with workspace_directory() as root:
            (root/"reports/paper").mkdir(parents=True)
            write_json(root/"experiments/plans/e_sync_coverage.json",{})
            references={}
            for split,count in (("training",30),("dev",20)):
                pairs=([study.CONTROL_PAIR]+[(f"train_{i}","w") for i in range(29)]
                       if split=="training" else [(f"dev_{i}","w") for i in range(20)])
                rows=[ScenarioResult(b,w,"e_sync",8640,-172.8,-.02,
                                     prod_regret=-144.,energy_cost=259.2,carbon=172.8)
                      for b,w in pairs]
                references[split]={(r.building,r.weather):r for r in rows}
                save(rows,root/f"experiments/runs/e_sync_coverage/{split}.json")

            def sandbox_path(*parts):
                path=Path(*parts)
                return root/path if path.parts and path.parts[0] in ("experiments","reports") else path

            class Worker:
                pid=123

                def __init__(self,command,**kwargs):
                    job=json.loads(Path(command[-1]).read_text())
                    task=job["task"]
                    policy=task["policy"]
                    factor=1.
                    if policy!="pi_cal_ctrl":
                        if secondary_wins:
                            factor=.98 if policy=="pi_cal_soft_fast" else 1.01
                        elif all_dev_fail or policy in ("pi_cal_soft_p","pi_cal_firm_p"):
                            factor=1.01
                        elif policy=="pi_cal_slow_i":
                            factor=.98 if task["split"]=="dev" else 1.01
                        else:
                            factor=.99
                    row=asdict(references[task["split"]][(task["building"],task["weather"])])
                    row["policy"]=policy
                    for field in ("total_reward","mean_reward","prod_regret","energy_cost","carbon"):
                        row[field]*=factor
                    write_json(job["result_path"],dict(task=task,fingerprint=job["fingerprint"],result=row))

                def poll(self):
                    return 0

            def package(command,**kwargs):
                key=command[command.index("--variant")+1]
                path=root/"experiments/submission"/key/"submission.ipynb"
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("{}")

            clock=itertools.count(0,100)
            with patch.object(study,"OUT",root/"runs"), \
                    patch.object(study,"Path",side_effect=sandbox_path), \
                    patch.object(study,"source_snapshot",return_value={}), \
                    patch.object(study.shutil,"which",return_value=None), \
                    patch.object(study.subprocess,"Popen",side_effect=Worker), \
                    patch.object(study.subprocess,"run",side_effect=package), \
                    patch.object(study.time,"monotonic",side_effect=lambda:next(clock)), \
                    patch.object(study.time,"sleep"), \
                    patch("psutil.virtual_memory") as memory, \
                    redirect_stdout(io.StringIO()):
                memory.return_value.available=8*2**30
                study.main()
            state=json.loads((root/"runs/status.json").read_text())
            self.assertEqual(state["status"],"complete")
            self.assertFalse((root/"runs/scheduler.lock").exists())
            for key in study.ALL_CANDIDATES if (all_dev_fail or secondary_wins) else study.CANDIDATES:
                self.assertEqual(state["counts"][key]["dev"],20)
            if secondary_wins:
                self.assertEqual(state["completed"],191)
                self.assertEqual(state["decision"]["selected"],"pi_cal_soft_fast")
                self.assertEqual(state["decision"]["calibration_batch"],2)
                self.assertTrue(Path(state["decision"]["submission_notebook"]).exists())
            elif all_dev_fail:
                self.assertEqual(state["completed"],161)
                self.assertNotIn("submission_notebook",state["decision"])
                self.assertFalse(state["decision"]["eligible_for_packaging"])
            else:
                self.assertEqual(state["completed"],141)
                decision=state["decision"]
                self.assertEqual(decision["selected"],"pi_cal_fast_i")
                self.assertFalse(decision["full_trials"]["pi_cal_slow_i"]["training"]["passes"])
                self.assertTrue(decision["full_trials"]["pi_cal_fast_i"]["training"]["passes"])
                self.assertTrue(Path(decision["submission_notebook"]).exists())
                self.assertTrue(Path(decision["submission_notebook"]).with_name("validation.json").exists())

    def test_failed_best_dev_candidate_falls_back_to_next_full_candidate(self):
        self.exercise(all_dev_fail=False)

    def test_no_notebook_is_exported_when_all_full_dev_means_regress(self):
        self.exercise(all_dev_fail=True)

    def test_secondary_batch_is_attempted_when_first_batch_cannot_win(self):
        self.exercise(all_dev_fail=False,secondary_wins=True)


if __name__ == "__main__":
    unittest.main()
