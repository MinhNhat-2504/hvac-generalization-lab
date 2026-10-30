"""Bounded zone-PI calibration with full-dev ranking and full-training fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import importlib.metadata

from hvaclab.coverage import digest, source_snapshot, valid_result, write_json

OUT = Path("experiments/runs/pi_calibration")
CONTROL_PAIR = ("building_train_7", "phoenix_3")
CANDIDATES = ("pi_cal_slow_i", "pi_cal_fast_i", "pi_cal_soft_p", "pi_cal_firm_p")
SECONDARY_CANDIDATES = ("pi_cal_soft_slow", "pi_cal_soft_fast", "pi_cal_firm_slow", "pi_cal_firm_fast")
ALL_CANDIDATES = CANDIDATES + SECONDARY_CANDIDATES
IMPROVEMENT_EPS = 1e-10



def run_study():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from hvaclab import runner
    from hvaclab.registry import _VARIANTS
    refs = {}
    for split in ("training", "dev"):
        refs[split] = {(r.building, r.weather): r for r in runner.load(
            Path("experiments/runs/e_sync_coverage") / f"{split}.json")}
        if len(refs[split]) != (30 if split == "training" else 20):
            raise ValueError("need complete e_sync reference coverage")
    dev_pairs = sorted(refs["dev"])
    train_pairs = sorted(refs["training"])
    plan = json.loads(Path("experiments/plans/e_sync_coverage.json").read_text())
    snapshot = source_snapshot(plan)
    snapshot["hvaclab/pi_calibration.py"] = digest(__file__)
    for path in ("hvaclab/submission/experimental.py", "hvaclab/submission/build.py"):
        snapshot[path] = digest(path)
    protocol = dict(source_sha256=snapshot,
                    reference_sha256={split: digest(Path("experiments/runs/e_sync_coverage") / f"{split}.json") for split in refs},
                    variants={p: _VARIANTS[p][1] for p in ("pi_cal_ctrl",)+ALL_CANDIDATES},
                    calibration_batches=[CANDIDATES,SECONDARY_CANDIDATES],
                    full_dev_pairs=dev_pairs, full_training_pairs=train_pairs,
                    control_pair=CONTROL_PAIR, steps=8640, max_workers=3,
                    available_memory_gib_per_launch=1.75, launch_spacing_seconds=20,
                    improvement_epsilon=IMPROVEMENT_EPS,
                    dependency_versions={p: importlib.metadata.version(p) for p in
                                         ("numpy","scipy","pandas","gin-config","protobuf")},
                    objective="First four one-coordinate PI neighbors; if none wins both complete cohorts, "
                              "try four fixed joint-gain corners. Integral gain times limit remains 1. "
                              "Evaluate all 20 dev cases per batch, rank by full-dev mean, and validate "
                              "30 training cases in order until one improves both full means. "
                              "No worst-case cutoff; report all regressions. No test-split access.")
    # JSON canonicalization also normalizes tuples for later resume comparison.
    protocol = json.loads(json.dumps(protocol))
    provenance = OUT / "protocol.json"
    if provenance.exists() and json.loads(provenance.read_text()) != protocol:
        raise ValueError("experiment sources or protocol changed; preserve results and use a new directory")
    write_json(provenance, protocol)
    fingerprint = digest(provenance)
    records, failed, active = {}, [], []
    memory_waiting = False
    stage, decision = "reference_control", {}

    def key(task):
        return task["split"], task["policy"], task["building"], task["weather"]

    def make_task(split, policy, pair):
        return dict(split=split, policy=policy, building=pair[0], weather=pair[1], steps=8640,
                    scenarios_dir="smart_buildings/training_scenarios" if split == "training"
                    else "smart_buildings/holdout_scenarios/dev")

    for path in sorted((OUT / "jobs").glob("*.result.json")):
        record = json.loads(path.read_text())
        if record["fingerprint"] != fingerprint:
            raise ValueError(f"stale result: {path}")
        if valid_result(record["result"], record["task"]):
            records[key(record["task"])] = record

    def statistics(split, policy, pairs):
        rows = [records[(split, policy, *p)]["result"] for p in pairs
                if (split, policy, *p) in records]
        if len(rows) != len(pairs):
            return None
        baseline = sum(refs[split][p].score for p in pairs) / len(pairs)
        score = sum(abs(r["mean_reward"]) for r in rows) / len(rows)
        worst_ratio = max(abs(r["mean_reward"])/refs[split][(r["building"],r["weather"])].score for r in rows)
        comparisons = [dict(scenario=r["building"]+"/"+r["weather"],
                            score=abs(r["mean_reward"]),
                            e_sync=refs[split][(r["building"],r["weather"])].score,
                            difference=abs(r["mean_reward"])-refs[split][(r["building"],r["weather"])].score)
                       for r in rows]
        return dict(n=len(rows), score=score, e_sync=baseline,
                    change_pct=100*(score/baseline-1), worst_ratio=worst_ratio,
                    improved=sum(c["difference"] < -IMPROVEMENT_EPS for c in comparisons),
                    regressed=sum(c["difference"] > IMPROVEMENT_EPS for c in comparisons),
                    largest_increases=sorted(comparisons,key=lambda c:c["difference"],reverse=True)[:3],
                    comfort=-.6*sum(r["prod_regret"]/r["steps"] for r in rows)/len(rows),
                    energy=.2*sum(r["energy_cost"]/r["steps"] for r in rows)/len(rows),
                    carbon=.2*sum(r["carbon"]/r["steps"] for r in rows)/len(rows),
                    passes=score < baseline-IMPROVEMENT_EPS)

    def checkpoint(status="running"):
        runner.save([runner.ScenarioResult(**r["result"]) for r in records.values()], OUT / "rollouts.json")
        state = dict(status=status, stage=stage, completed=len(records), failed=failed,
                     waiting_for_memory=memory_waiting,
                     active=[item[0] for item in active], decision=decision,
                     counts={p: {split: sum(1 for k in records if k[0]==split and k[1]==p)
                                 for split in ("dev","training")} for p in ALL_CANDIDATES},
                     updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(OUT / "status.json", state)
        lines = ["# Full-development zone-PI calibration", "", f"Status: {status}; stage: {stage}.", "",
                 "All scores use complete 8,640-step episodes against matched e_sync references.", "",
                 "| Cohort | Variant | n | e_sync | Candidate | Change |", "|---|---|---:|---:|---:|---:|"]
        tex = ["\\paragraph{Automatically recorded closed-loop results.}",
               "Status: \\texttt{" + status.replace("_", "\\_") + "}; stage: \\texttt{" + stage.replace("_", "\\_") + "}.",
               "All entries below require complete 8,640-step episodes on matching scenarios.",
               "\\begin{center}\\small", "\\begin{tabular}{llrrrr}\\toprule",
               "Cohort & Variant & $n$ & e\\_sync & Candidate & Change \\\\\\midrule"]
        cohorts = [("full dev", "dev", dev_pairs), ("full training", "training", train_pairs)]
        for label, split, pairs in cohorts:
            for policy in ALL_CANDIDATES:
                s = statistics(split, policy, pairs)
                if s is None:
                    continue
                lines.append(f"| {label} | {policy} | {s['n']} | {s['e_sync']:.8f} | {s['score']:.8f} | {s['change_pct']:+.2f}% |")
                tex.append(f"{label} & \\texttt{{{policy.replace('_', chr(92)+'_')}}} & {s['n']} & "
                           f"{s['e_sync']:.8f} & {s['score']:.8f} & {s['change_pct']:+.2f}\\% \\\\")
        tex += ["\\bottomrule\\end{tabular}\\end{center}",
                "These are complete matched cohorts. Dev is repeatedly used model-selection data, not untouched validation. Individual regressions are reported without a hard cutoff.",
                "Completed new rollouts: " + str(len(records)) + ". No candidate replaces Version 4 automatically."]
        if "unchanged_pi_full_episode_parity" in decision:
            control = records[("training", "pi_cal_ctrl", *CONTROL_PAIR)]["result"]
            tex.append("The unchanged-PI full-episode control score is "
                       f"${abs(control['mean_reward']):.8f}$; exact-reproduction gate: "
                       + ("passed." if decision["unchanged_pi_full_episode_parity"] else "failed."))
        if "stop_reason" in decision:
            tex.append("Experiment decision: " + decision["stop_reason"] + ".")
        if "selected" in decision:
            tex.append("The selected candidate is \\texttt{" + decision["selected"].replace("_", "\\_") + "}.")
        lines += ["", "Decision:", "", "```json", json.dumps(decision, indent=2), "```", "",
                  "Earlier paper sections and the e_sync public score remain unchanged. "
                  "Selection optimizes full-cohort mean score. Full-training failures remain recorded while the next dev-ranked candidate is tried."]
        (Path("reports/PI_CALIBRATION_REVIEW.md")).write_text("\n".join(lines), encoding="utf-8")
        (Path("reports/paper/pi_calibration_results.tex")).write_text("\n".join(tex)+"\n", encoding="utf-8")

    def compile_paper():
        executable = shutil.which("pdflatex")
        if executable:
            with (OUT / f"paper_{stage}.log").open("w", encoding="utf-8") as log:
                for _ in range(2):
                    result = subprocess.run([executable, "-no-shell-escape", "-interaction=nonstopmode",
                                             "-halt-on-error", "main.tex"], cwd="reports/paper", stdout=log,
                                            stderr=subprocess.STDOUT)
                    if result.returncode:
                        print("Paper PDF build failed; source and results saved. See build log.", flush=True)
                        break

    def execute(tasks):
        nonlocal memory_waiting
        pending = [task for task in tasks if key(task) not in records]
        heartbeat = time.monotonic()
        next_launch = 0.
        while pending or active:
            while pending and len(active) < 3:
                import psutil
                # Allow the new interpreter to allocate before checking RAM
                # for another one; avoid two simultaneous initialization peaks.
                if active and time.monotonic() < next_launch:
                    break
                required_gib = 1.75
                if psutil.virtual_memory().available < required_gib * 2**30:
                    if not memory_waiting:
                        memory_waiting = True
                        print(f"WAIT RAM: need {required_gib:.2f} GiB available before another rollout", flush=True)
                        checkpoint()
                    break
                memory_waiting = False
                task = pending.pop(0)
                stem = "__".join(key(task))
                descriptor = OUT / "jobs" / (stem + ".job.json")
                result_path = descriptor.with_name(stem + ".result.json")
                write_json(descriptor, dict(task=task, fingerprint=fingerprint,
                                           result_path=str(result_path), source_sha256=snapshot))
                log = descriptor.with_suffix(".log").open("w", encoding="utf-8")
                process = subprocess.Popen([sys.executable, "-m", "hvaclab.coverage", "--job", str(descriptor)],
                                           stdout=log, stderr=subprocess.STDOUT)
                active.append((task, process, log, result_path))
                next_launch = time.monotonic() + 20.
                print(f"START {key(task)} pid={process.pid}", flush=True)
                checkpoint()
            changed = False
            for item in list(active):
                task, process, log, path = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                log.close()
                record = json.loads(path.read_text()) if path.exists() else None
                if code == 0 and record and record["fingerprint"] == fingerprint and record["task"] == task and valid_result(record["result"],task):
                    records[key(task)] = record
                    print(f"DONE {key(task)} score={abs(record['result']['mean_reward']):.8f}", flush=True)
                else:
                    failed.append(dict(task=task, exit_code=code, result_path=str(path)))
                with (OUT / "attempts.jsonl").open("a", encoding="utf-8") as history:
                    history.write(json.dumps(dict(task=task, exit_code=code, record=record))+"\n")
                changed = True
            if changed or time.monotonic()-heartbeat >= 60:
                checkpoint()
                heartbeat = time.monotonic()
                print(f"PROGRESS stage={stage}; complete={len(records)}; active={len(active)}; failures={len(failed)}", flush=True)
            if active or pending:
                time.sleep(2)
        if failed:
            checkpoint("failed")
            raise RuntimeError("rollout failure: inspect per-job logs and resume")

    checkpoint()
    if "--prepare-only" in sys.argv:
        checkpoint("prepared")
        compile_paper()
        return
    try:
        execute([make_task("training","pi_cal_ctrl",CONTROL_PAIR)])
        control=records[("training","pi_cal_ctrl",*CONTROL_PAIR)]["result"]
        parity=abs(control["mean_reward"]-refs["training"][CONTROL_PAIR].mean_reward)<=1e-10
        decision["unchanged_pi_full_episode_parity"]=parity
        if not parity:
            raise RuntimeError("V4 reference-control mismatch; stop calibration")
        decision["full_dev_calibration"]={}
        decision["ranked_candidates"]={}
        decision["full_trials"]={}
        for batch_number,batch in enumerate((CANDIDATES,SECONDARY_CANDIDATES),1):
            decision["calibration_batch"]=batch_number
            stage="full_dev_calibration"
            checkpoint()
            compile_paper()
            execute([make_task("dev",p,pair) for pair in dev_pairs for p in batch])
            decision["full_dev_calibration"].update({p: statistics("dev",p,dev_pairs) for p in batch})
            eligible=sorted((p for p in batch if decision["full_dev_calibration"][p]["passes"]),
                            key=lambda p: decision["full_dev_calibration"][p]["score"])
            decision["ranked_candidates"][str(batch_number)]=eligible
            for selected in eligible:
                stage="full_training_validation"
                decision["evaluating_candidate"]=selected
                checkpoint()
                compile_paper()
                execute([make_task("training",selected,pair) for pair in train_pairs])
                training=statistics("training",selected,train_pairs)
                dev=decision["full_dev_calibration"][selected]
                decision["full_trials"][selected]=dict(training=training,dev=dev)
                decision["full_training"],decision["full_dev"]=training,dev
                if not training["passes"]:
                    checkpoint()
                    continue
                decision["selected"]=selected
                decision["eligible_for_packaging"]=True
                stage="packaging"
                checkpoint()
                compile_paper()
                with (OUT/"packaging.log").open("w",encoding="utf-8") as log:
                    subprocess.run([sys.executable,"-u","-m","hvaclab.submission.experimental",
                                    "--variant",selected,"--verify"],stdout=log,stderr=subprocess.STDOUT,check=True)
                notebook=Path("experiments/submission")/selected/"submission.ipynb"
                decision["submission_notebook"]=notebook.as_posix()
                params=_VARIANTS[selected][1]
                description=(f"e_sync/V4 + zone-PI calibration: Kp={params['kp']}, Ki={params['ki']}, "
                             f"integral limit={params['integral_limit']}; maximum integral contribution retained at 1. "
                             "AHU/pump coordination, flat bands and 288-step cooling hold unchanged. "
                             "Lower mean on matched full training/dev cohorts; standalone action parity verified. "
                             "No public-score claim.")
                (notebook.parent/"SUBMISSION_DESCRIPTION.txt").write_text(description,encoding="utf-8")
                validation=dict(protocol_sha256=fingerprint,validation_scope="complete local cohorts, not Kaggle private score",
                                training=training,dev=dev,notebook_sha256=digest(notebook))
                write_json(notebook.parent/"validation.json",validation)
                break
            if "submission_notebook" in decision:
                break
        if "submission_notebook" not in decision:
            decision["eligible_for_packaging"]=False
            decision["stop_reason"]="no tested PI neighbor improved both full means; keep V4"
        stage="finished"
        checkpoint("complete")
        compile_paper()
    except BaseException as exc:
        import psutil
        for _,process,log,_ in list(active):
            try:
                parent=psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=10)
            except (psutil.NoSuchProcess,psutil.AccessDenied,OSError,subprocess.TimeoutExpired):
                pass
            log.close()
        active.clear()
        checkpoint("interrupted" if isinstance(exc,KeyboardInterrupt) else "failed")
        compile_paper()
        raise


class RunLock:
    def __init__(self,path):
        self.path=Path(path)
        self.owner=None

    @staticmethod
    def _owner_is_live(owner):
        import psutil
        try:
            process=psutil.Process(owner["pid"])
            return abs(process.create_time()-owner["process_created_at"])<.01
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return True

    def __enter__(self):
        import psutil
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.owner=dict(pid=os.getpid(),process_created_at=psutil.Process().create_time())
        for _ in range(2):
            try:
                with self.path.open("x",encoding="utf-8") as stream:
                    json.dump(self.owner,stream)
                return self
            except FileExistsError:
                previous=self.path.read_text(encoding="utf-8")
                owner=json.loads(previous)
                if self._owner_is_live(owner):
                    raise RuntimeError(f"calibration scheduler already running: pid {owner['pid']}")
                if self.path.read_text(encoding="utf-8")!=previous:
                    raise RuntimeError("another scheduler is acquiring the lock")
                self.path.unlink()
        raise RuntimeError("could not acquire scheduler lock")

    def __exit__(self,exc_type,exc,tb):
        if self.path.exists() and json.loads(self.path.read_text(encoding="utf-8"))==self.owner:
            self.path.unlink()


def main():
    with RunLock(OUT/"scheduler.lock"):
        run_study()


if __name__ == "__main__":
    main()

