from contextlib import redirect_stdout
from dataclasses import asdict
import io
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import psutil

from hvaclab import parallel_valve_calibration as study
from hvaclab.coverage import write_json, digest
from hvaclab.runner import ScenarioResult, save
from test_coverage import workspace_directory


class CalibrationWorkflowTest(unittest.TestCase):
    def exercise(self,all_dev_fail,v4_stronger=False,predecessor_stronger=False,family="parallel_valve",concurrent_stronger=False):
        from hvaclab.parallel_valve_entry import SPECS
        spec=SPECS[family]
        control=spec["control"]
        candidates=tuple(k for k in spec["arms"] if k!=control)
        with workspace_directory() as root:
            (root/"reports/paper").mkdir(parents=True)
            write_json(root/"experiments/plans/e_sync_coverage.json",{})
            from hvaclab.parallel_valve_entry import register
            register()
            write_json(root/"experiments/runs/persistence_calibration/status.json",
                       dict(status="complete",decision={}))
            save([],root/"experiments/runs/persistence_calibration/rollouts.json")
            write_json(root/'experiments/runs/damper_persistence/status.json',dict(status='complete',decision={}))
            save([],root/'experiments/runs/damper_persistence/rollouts.json')
            references={}
            curve_references={}
            curve_rows=[]
            for split,count in (("training",30),("dev",20)):
                pairs=([study.CONTROL_PAIR]+[(f"train_{i}","w") for i in range(29)]
                       if split=="training" else [(f"dev_{i}","w") for i in range(20)])
                rows=[ScenarioResult(b,w,"e_sync",8640,-172.8,-.02,
                                     prod_regret=-144.,energy_cost=259.2,carbon=172.8)
                      for b,w in pairs]
                references[split]={(r.building,r.weather):r for r in rows}
                curve_references[split]={}
                for row in rows:
                    curve=asdict(row)
                    curve["policy"]="valve8_gentle"
                    for field in ("total_reward","mean_reward","prod_regret","energy_cost","carbon"):
                        curve[field]*=.99
                        if v4_stronger:setattr(row,field,getattr(row,field)*.95)
                    curve_row=ScenarioResult(**curve)
                    curve_rows.append(curve_row)
                    curve_references[split][(row.building,row.weather)]=curve_row
                save(rows,root/f"experiments/runs/e_sync_coverage/{split}.json")
            save(curve_rows,root/"experiments/runs/reheat_curve_calibration/rollouts.json")
            if concurrent_stronger:
                extra=[]
                for r in curve_rows:
                    value=asdict(r)
                    value['policy']='concurrent_winner'
                    for field in ('mean_reward','total_reward','prod_regret','energy_cost','carbon'):
                        value[field]*=.95
                    extra.append(ScenarioResult(**value))
                write_json(root/'experiments/runs/damper_persistence/status.json',
                           dict(status='complete',decision=dict(selected='concurrent_winner')))
                save(extra,root/'experiments/runs/damper_persistence/rollouts.json')
            fixed=[]
            earlier=[]
            for row in curve_rows:
                value=asdict(row)
                value['policy']='persist_432'
                fixed.append(ScenarioResult(**value))
                value=asdict(row)
                value['policy']='fake_prior'
                for field in ('mean_reward','total_reward','prod_regret','energy_cost','carbon'):
                    value[field]*=.95 if predecessor_stronger else 1.02
                earlier.append(ScenarioResult(**value))
            write_json(root/'experiments/runs/persistence_calibration/status.json',
                       dict(status='complete',decision=dict(selected='persist_432')))
            save(fixed,root/'experiments/runs/persistence_calibration/rollouts.json')
            write_json(root/'experiments/runs/hold_release_calibration/status.json',
                       dict(status='complete',decision=dict(selected='fake_prior')))
            save(earlier,root/'experiments/runs/hold_release_calibration/rollouts.json')
            for dependency in spec['dependencies']:
                folder=root/'experiments/runs'/dependency
                if (folder/'status.json').exists():continue
                if dependency=='heating_damper':
                    rows=[ScenarioResult(**{**asdict(r),'policy':'damper11_less'}) for r in fixed]
                    write_json(folder/'status.json',dict(status='complete',decision=dict(selected='damper11_less')))
                    save(rows,folder/'rollouts.json')
                else:
                    write_json(folder/'status.json',dict(status='complete',decision={}))
                    save([],folder/'rollouts.json')


            def sandbox_path(*parts):
                path=Path(*parts)
                return root/path if path.parts and path.parts[0] in ("experiments","reports") else path

            class Worker:
                pid=123
                live=0
                peak=0

                def __init__(self,command,**kwargs):
                    Worker.live+=1
                    Worker.peak=max(Worker.peak,Worker.live)
                    self.finished=False
                    job=json.loads(Path(command[-1]).read_text())
                    task=job["task"]
                    policy=task["policy"]
                    factor=1.
                    if policy!=control:
                        if all_dev_fail or policy==candidates[2]:
                            factor=1.01
                        elif policy==candidates[0]:
                            factor=.98 if task["split"]=="dev" else 1.01
                        else:
                            factor=.99
                    source=curve_references
                    row=asdict(source[task["split"]][(task["building"],task["weather"])])
                    row["policy"]=policy
                    for field in ("total_reward","mean_reward","prod_regret","energy_cost","carbon"):
                        row[field]*=factor
                    write_json(job["result_path"],dict(task=task,fingerprint=job["fingerprint"],result=row))

                def poll(self):
                    if not self.finished:
                        Worker.live-=1
                        self.finished=True
                    return 0

            def package(command,**kwargs):
                key=command[command.index("--variant")+1]
                path=root/"experiments/submission"/key/"submission.ipynb"
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("{}")

            clock=itertools.count(0,100)
            with patch.multiple(study,STUDY=family,CONTROL=control,CANDIDATES=candidates,ALL_CANDIDATES=candidates), \
                    patch.object(study,"OUT",root/"runs"), \
                    patch.object(study,"Path",side_effect=sandbox_path), \
                    patch.object(study,"source_snapshot",return_value={}), \
                    patch.object(study.shutil,"which",return_value=None), \
                    patch.object(study.subprocess,"Popen",side_effect=Worker), \
                    patch.object(study.subprocess,"run",side_effect=package), \
                    patch.object(study.time,"monotonic",side_effect=lambda:next(clock)), \
                    patch.object(study.time,"sleep"), \
                    patch("psutil.virtual_memory") as memory, \
                    redirect_stdout(io.StringIO()):
                memory.return_value.available=2.6*2**30
                study.run_study()
            self.assertEqual(Worker.peak,1)
            state=json.loads((root/"runs/status.json").read_text())
            self.assertEqual(state["status"],"complete")
            self.assertFalse((root/"runs/scheduler.lock").exists())
            for key in study.CANDIDATES:
                self.assertEqual(state["counts"][key]["dev"],20)
            if concurrent_stronger:
                self.assertEqual(state['completed'],121)
                self.assertFalse(state['decision']['eligible_for_packaging'])
                self.assertNotIn('submission_notebook',state['decision'])
            elif all_dev_fail or v4_stronger or predecessor_stronger:
                self.assertEqual(state["completed"],61)
                self.assertNotIn("submission_notebook",state["decision"])
                self.assertFalse(state["decision"]["eligible_for_packaging"])
            else:
                self.assertEqual(state["completed"],121)
                decision=state["decision"]
                self.assertEqual(decision["selected"],candidates[1])
                self.assertFalse(decision["full_trials"][candidates[0]]["training"]["passes"])
                self.assertTrue(decision["full_trials"][candidates[1]]["training"]["passes"])
                self.assertTrue(Path(decision["submission_notebook"]).exists())
                self.assertTrue(Path(decision["submission_notebook"]).with_name("validation.json").exists())

    def test_failed_best_dev_candidate_falls_back_to_next_full_candidate(self):
        self.exercise(all_dev_fail=False)

    def test_no_notebook_is_exported_when_all_full_dev_means_regress(self):
        self.exercise(all_dev_fail=True)

    def test_improvement_over_submitted_curve_without_beating_v4_does_not_export(self):
        self.exercise(all_dev_fail=False,v4_stronger=True)

    def test_prior_winner_is_a_required_comparator(self):
        self.exercise(False,predecessor_stronger=True)

    def test_concurrent_winner_vetoes_otherwise_eligible_candidate(self):
        self.exercise(False,concurrent_stronger=True)



if __name__=='__main__':unittest.main()
