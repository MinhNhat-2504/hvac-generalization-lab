from contextlib import redirect_stdout
from dataclasses import asdict
import io
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import psutil

from hvaclab import split_geometry_calibration as study
from hvaclab.coverage import write_json, digest
from hvaclab.runner import ScenarioResult, save
from test_coverage import workspace_directory


class CalibrationWorkflowTest(unittest.TestCase):
    def exercise(self,all_dev_fail,v4_stronger=False,v5_stronger=False):
        with workspace_directory() as root:
            (root/"reports/paper").mkdir(parents=True)
            write_json(root/"experiments/plans/e_sync_coverage.json",{})
            write_json(root/"experiments/plans/geometry_reference.json", dict(
                reference_area_cv=2484., reference_uses_training_only=True,
                buildings=[dict(component_and_sdk_mapping_verified=True,
                                zone_map="hvaclab/policies/geometry_pi.py",
                                zone_map_sha256=digest("hvaclab/policies/geometry_pi.py")) for _ in range(10)],
                verification_script_sha256=digest("hvaclab/geometry_reference.py"),
                component_policy_sha256=digest("hvaclab/policies/geometry_pi.py")))
            references={}
            v5_rows=[]
            for split,count in (("training",30),("dev",20)):
                pairs=([study.CONTROL_PAIR]+[(f"train_{i}","w") for i in range(29)]
                       if split=="training" else [(f"dev_{i}","w") for i in range(20)])
                rows=[ScenarioResult(b,w,"e_sync",8640,-172.8,-.02,
                                     prod_regret=-144.,energy_cost=259.2,carbon=172.8)
                      for b,w in pairs]
                references[split]={(r.building,r.weather):r for r in rows}
                for row in rows:
                    v5=asdict(row)
                    v5["policy"]="geo6_threequarter"
                    v5_rows.append(ScenarioResult(**v5))
                    for field in ("total_reward","mean_reward","prod_regret","energy_cost","carbon"):
                        setattr(row,field,getattr(row,field)*(.97 if v4_stronger else 1.1))
                save(rows,root/f"experiments/runs/e_sync_coverage/{split}.json")
            save(v5_rows,root/"experiments/runs/geometry_refinement/rollouts.json")
            older_rows=[]
            for row in v5_rows:
                old=asdict(row)
                old["policy"]="geo_pi_sqrt"
                for field in ("total_reward","mean_reward","prod_regret","energy_cost","carbon"):
                    old[field]*=.97 if v5_stronger else 1.05
                older_rows.append(ScenarioResult(**old))
            save(older_rows,root/"experiments/runs/geometry_calibration/rollouts.json")
            references={split:{(r.building,r.weather):next(v for v in v5_rows if
                         (v.building,v.weather)==(r.building,r.weather)) for r in mapping.values()}
                         for split,mapping in references.items()}

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
                    if policy!="geo7_ctrl":
                        if all_dev_fail or policy=="geo7_soft_p":
                            factor=1.01
                        elif policy=="geo7_soft_i":
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
            for key in study.CANDIDATES:
                self.assertEqual(state["counts"][key]["dev"],20)
            if all_dev_fail or v4_stronger or v5_stronger:
                self.assertEqual(state["completed"],61)
                self.assertNotIn("submission_notebook",state["decision"])
                self.assertFalse(state["decision"]["eligible_for_packaging"])
            else:
                self.assertEqual(state["completed"],121)
                decision=state["decision"]
                self.assertEqual(decision["selected"],"geo7_strong_i")
                self.assertFalse(decision["full_trials"]["geo7_soft_i"]["training"]["passes"])
                self.assertTrue(decision["full_trials"]["geo7_strong_i"]["training"]["passes"])
                self.assertLess(decision["full_training"]["score"],decision["full_training"]["v6"])
                self.assertLess(decision["full_training"]["score"],decision["full_training"]["e_sync"])
                self.assertTrue(Path(decision["submission_notebook"]).exists())
                self.assertTrue(Path(decision["submission_notebook"]).with_name("validation.json").exists())

    def test_failed_best_dev_candidate_falls_back_to_next_full_candidate(self):
        self.exercise(all_dev_fail=False)

    def test_no_notebook_is_exported_when_all_full_dev_means_regress(self):
        self.exercise(all_dev_fail=True)

    def test_improving_v6_without_beating_v4_does_not_export(self):
        self.exercise(all_dev_fail=False,v4_stronger=True)
    def test_improving_v6_without_beating_v5_does_not_export(self):
        self.exercise(all_dev_fail=False,v5_stronger=True)


if __name__ == "__main__":
    unittest.main()
