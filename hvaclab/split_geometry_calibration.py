"""V7 split geometry sensitivities against V6, V5 and V4 references."""
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
from hvaclab.pi_calibration import RunLock

OUT = Path("experiments/runs/split_geometry_calibration")
CONTROL_PAIR = ("building_train_7", "phoenix_3")
CANDIDATES = ("geo7_soft_i", "geo7_strong_i", "geo7_soft_p")
ALL_CANDIDATES = CANDIDATES
IMPROVEMENT_EPS = 1e-10



def run_study():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from hvaclab import runner
    from hvaclab.registry import _VARIANTS
    v4_refs = {}
    refs = {}
    v6_path = Path("experiments/runs/geometry_refinement/rollouts.json")
    v6_rows = [r for r in runner.load(v6_path) if r.policy == "geo6_threequarter"]
    if len(v6_rows) != 50 or len({(r.building,r.weather) for r in v6_rows}) != 50:
        raise ValueError("need 50 unique complete V6 reference episodes")
    for row in v6_rows:
        task=dict(building=row.building,weather=row.weather,policy="geo6_threequarter",steps=8640)
        if not valid_result(vars(row),task):
            raise ValueError("invalid V6 reference episode")
    for split in ("training", "dev"):
        v4_refs[split] = {(r.building, r.weather): r for r in runner.load(
            Path("experiments/runs/e_sync_coverage") / f"{split}.json")}
        refs[split] = {(r.building,r.weather):r for r in v6_rows
                       if (r.building,r.weather) in v4_refs[split]}
        if len(refs[split]) != (30 if split == "training" else 20):
            raise ValueError("need complete matched V4 and V6 reference coverage")
        if any(not valid_result(vars(r),dict(building=r.building,weather=r.weather,
                                            policy="e_sync",steps=8640)) for r in v4_refs[split].values()):
            raise ValueError("invalid V4 reference episode")
    if set(refs["training"]) & set(refs["dev"]):
        raise ValueError("reference cohorts overlap")
    older_path=Path("experiments/runs/geometry_calibration/rollouts.json")
    older_rows=[r for r in runner.load(older_path) if r.policy=="geo_pi_sqrt"]
    if len(older_rows)!=50 or len({(r.building,r.weather) for r in older_rows})!=50:
        raise ValueError("need 50 unique V5 reference episodes")
    for row in older_rows:
        if not valid_result(vars(row),dict(building=row.building,weather=row.weather,
                                         policy="geo_pi_sqrt",steps=8640)):
            raise ValueError("invalid V5 reference episode")
    older_refs={split:{(r.building,r.weather):r for r in older_rows
                       if (r.building,r.weather) in refs[split]} for split in refs}
    if any(len(older_refs[split])!=len(refs[split]) for split in refs):
        raise ValueError("V5 references do not match cohorts")
    dev_pairs = sorted(refs["dev"])
    train_pairs = sorted(refs["training"])
    plan = json.loads(Path("experiments/plans/e_sync_coverage.json").read_text())
    snapshot = source_snapshot(plan)
    snapshot["hvaclab/split_geometry_calibration.py"] = digest(__file__)
    for path in ("hvaclab/submission/experimental.py", "hvaclab/submission/build.py"):
        snapshot[path] = digest(path)
    geometry_path=Path("experiments/plans/geometry_reference.json")
    geometry=json.loads(geometry_path.read_text())
    if (geometry["reference_area_cv"]!=2484. or not geometry["reference_uses_training_only"]
            or len(geometry["buildings"])!=10
            or not all(b["component_and_sdk_mapping_verified"] for b in geometry["buildings"])):
        raise ValueError("geometry reference/mapping verification missing")
    if geometry["component_policy_sha256"]!=digest("hvaclab/policies/geometry_pi.py"):
        raise ValueError("geometry policy changed since mapping verification")
    if geometry["verification_script_sha256"]!=digest("hvaclab/geometry_reference.py"):
        raise ValueError("geometry verifier changed since mapping verification")
    snapshot["hvaclab/geometry_reference.py"]=digest("hvaclab/geometry_reference.py")
    for building in geometry["buildings"]:
        if digest(building["zone_map"])!=building["zone_map_sha256"]:
            raise ValueError("zone map changed since mapping verification")
        snapshot[building["zone_map"]]=building["zone_map_sha256"]
    snapshot["hvaclab/pi_calibration.py"]=digest("hvaclab/pi_calibration.py")
    protocol = dict(v6_reference_sha256=digest(v6_path), v5_reference_sha256=digest(older_path), geometry_reference_sha256=digest(geometry_path),source_sha256=snapshot,
                    reference_sha256={split: digest(Path("experiments/runs/e_sync_coverage") / f"{split}.json") for split in refs},
                    variants={p: _VARIANTS[p][1] for p in ("geo7_ctrl",)+ALL_CANDIDATES},
                    calibration_batches=[CANDIDATES],
                    full_dev_pairs=dev_pairs, full_training_pairs=train_pairs,
                    control_pair=CONTROL_PAIR, steps=8640, max_workers=3,
                    available_memory_gib_per_launch=1.75, launch_spacing_seconds=20,
                    available_disk_gib_per_launch=1.0,
                    improvement_epsilon=IMPROVEMENT_EPS,
                    dependency_versions={p: importlib.metadata.version(p) for p in
                                         ("numpy","scipy","pandas","gin-config","protobuf")},
                    objective="V7: separate P and I geometry exponents (0.75,0.5), (0.75,1.0), (0.5,0.75); metadata only. "
                              "reference area is training-only median, clipping scales to [0.5,2]. "
                              "Evaluate all 20 dev cases for all three; rank by full-dev mean and "
                              "check all 30 training cases in order until both cohort means beat V6, V5 and V4. "
                              "No worst-case cutoff. No test-split access.")

    # JSON canonicalization also normalizes tuples for later resume comparison.
    protocol = json.loads(json.dumps(protocol))
    provenance = OUT / "protocol.json"
    if provenance.exists() and json.loads(provenance.read_text()) != protocol:
        raise ValueError("experiment sources or protocol changed; preserve results and use a new directory")
    write_json(provenance, protocol)
    fingerprint = digest(provenance)
    records, failed, active = {}, [], []
    memory_waiting = False
    disk_waiting = False
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
        v4 = sum(v4_refs[split][p].score for p in pairs) / len(pairs)
        older = sum(older_refs[split][p].score for p in pairs) / len(pairs)
        worst_ratio = max(abs(r["mean_reward"])/refs[split][(r["building"],r["weather"])].score for r in rows)
        comparisons = [dict(scenario=r["building"]+"/"+r["weather"],
                            score=abs(r["mean_reward"]),
                            v6=refs[split][(r["building"],r["weather"])].score,
                            difference=abs(r["mean_reward"])-refs[split][(r["building"],r["weather"])].score)
                       for r in rows]
        return dict(n=len(rows), score=score, v6=baseline, v5=older, e_sync=v4,
                    change_vs_v5_pct=100*(score/older-1),
                    change_vs_v4_pct=100*(score/v4-1),
                    change_pct=100*(score/baseline-1), worst_ratio=worst_ratio,
                    improved=sum(c["difference"] < -IMPROVEMENT_EPS for c in comparisons),
                    regressed=sum(c["difference"] > IMPROVEMENT_EPS for c in comparisons),
                    largest_increases=sorted(comparisons,key=lambda c:c["difference"],reverse=True)[:3],
                    comfort=-.6*sum(r["prod_regret"]/r["steps"] for r in rows)/len(rows),
                    energy=.2*sum(r["energy_cost"]/r["steps"] for r in rows)/len(rows),
                    carbon=.2*sum(r["carbon"]/r["steps"] for r in rows)/len(rows),
                    passes=score < min(baseline,older,v4)-IMPROVEMENT_EPS)

    def checkpoint(status="running"):
        runner.save([runner.ScenarioResult(**r["result"]) for r in records.values()], OUT / "rollouts.json")
        state = dict(status=status, stage=stage, completed=len(records), failed=failed,
                     waiting_for_memory=memory_waiting,
                     waiting_for_disk=disk_waiting,
                     active=[item[0] for item in active], decision=decision,
                     counts={p: {split: sum(1 for k in records if k[0]==split and k[1]==p)
                                 for split in ("dev","training")} for p in ALL_CANDIDATES},
                     updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(OUT / "status.json", state)
        lines = ["# V7 split geometry sensitivity versus V6, V5 and V4", "", f"Status: {status}; stage: {stage}.", "",
                 "All scores use complete 8,640-step episodes against matched V6 references (V5 and V4 also checked).", "",
                 "| Cohort | Variant | n | V6 | Candidate | Change vs V6 |", "|---|---|---:|---:|---:|---:|"]
        tex = ["\\paragraph{Automatically recorded closed-loop results.}",
               "Status: \\texttt{" + status.replace("_", "\\_") + "}; stage: \\texttt{" + stage.replace("_", "\\_") + "}.",
               "All entries below require complete 8,640-step episodes on matching scenarios.",
               "\\begin{center}\\small", "\\begin{tabular}{llrrrr}\\toprule",
               "Cohort & Variant & $n$ & V6 & Candidate & Change \\\\\\midrule"]
        cohorts = [("full dev", "dev", dev_pairs), ("full training", "training", train_pairs)]
        for label, split, pairs in cohorts:
            for policy in ALL_CANDIDATES:
                s = statistics(split, policy, pairs)
                if s is None:
                    continue
                lines.append(f"| {label} | {policy} | {s['n']} | {s['v6']:.8f} | {s['score']:.8f} | {s['change_pct']:+.2f}% |")
                tex.append(f"{label} & \\texttt{{{policy.replace('_', chr(92)+'_')}}} & {s['n']} & "
                           f"{s['v6']:.8f} & {s['score']:.8f} & {s['change_pct']:+.2f}\\% \\\\")
        tex += ["\\bottomrule\\end{tabular}\\end{center}",
                "These are complete matched cohorts. Dev is repeatedly used model-selection data, not untouched validation. Individual regressions are reported without a hard cutoff.",
                "Completed new rollouts: " + str(len(records)) + ". V6 remains the latest submitted candidate; organizer improvement is not established."]
        if "unchanged_v6_full_episode_parity" in decision:
            control = records[("training", "geo7_ctrl", *CONTROL_PAIR)]["result"]
            tex.append("The unchanged-V6 full-episode control score is "
                       f"${abs(control['mean_reward']):.8f}$; exact-reproduction gate: "
                       + ("passed." if decision["unchanged_v6_full_episode_parity"] else "failed."))
        if "stop_reason" in decision:
            tex.append("Experiment decision: " + decision["stop_reason"] + ".")
        if "selected" in decision:
            tex.append("The selected candidate is \\texttt{" + decision["selected"].replace("_", "\\_") + "}.")
        lines += ["", "Decision:", "", "```json", json.dumps(decision, indent=2), "```", "",
                  "Earlier paper sections and V4/V5/V6 artifacts remain unchanged. Change percentages use V6; JSON also reports V5 and V4. "
                  "Selection requires lower full-cohort means than V6, V5 and V4. Full-training failures remain recorded while the next dev-ranked candidate is tried."]
        (Path("reports/SPLIT_GEOMETRY_CALIBRATION_REVIEW.md")).write_text("\n".join(lines), encoding="utf-8")
        (Path("reports/paper/split_geometry_calibration_results.tex")).write_text("\n".join(tex)+"\n", encoding="utf-8")

    def compile_paper():
        executable = shutil.which("pdflatex")
        if executable:
            build_dir=(OUT/"paper_build").resolve()
            build_dir.mkdir(parents=True,exist_ok=True)
            with (OUT / f"paper_{stage}.log").open("w", encoding="utf-8") as log:
                for _ in range(2):
                    result = subprocess.run([executable, "-no-shell-escape", "-interaction=nonstopmode",
                                             "-halt-on-error", "-output-directory",str(build_dir),
                                             "main.tex"], cwd="reports/paper", stdout=log,
                                            stderr=subprocess.STDOUT)
                    if result.returncode:
                        print("Paper PDF build failed; source and results saved. See build log.", flush=True)
                        return
            # Keep the previous readable PDF if TeX fails or a drive fills.
            temporary=Path("reports/paper/main.pdf.tmp")
            shutil.copyfile(build_dir/"main.pdf",temporary)
            temporary.replace(Path("reports/paper/main.pdf"))

    def execute(tasks):
        nonlocal memory_waiting,disk_waiting
        pending = [task for task in tasks if key(task) not in records]
        heartbeat = time.monotonic()
        next_launch = 0.
        while pending or active:
            while pending and len(active) < 3:
                import psutil
                if shutil.disk_usage(OUT).free < 2**30:
                    if not disk_waiting:
                        disk_waiting=True
                        print("WAIT DISK: need 1 GiB free before another rollout",flush=True)
                        checkpoint()
                    break
                disk_waiting=False
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
        execute([make_task("training","geo7_ctrl",CONTROL_PAIR)])
        control=records[("training","geo7_ctrl",*CONTROL_PAIR)]["result"]
        parity=abs(control["mean_reward"]-refs["training"][CONTROL_PAIR].mean_reward)<=1e-10
        decision["unchanged_v6_full_episode_parity"]=parity
        if not parity:
            raise RuntimeError("V6 reference-control mismatch; stop calibration")
        decision["full_dev_calibration"]={}
        decision["ranked_candidates"]={}
        decision["full_trials"]={}
        for batch_number,batch in enumerate((CANDIDATES,),1):
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
                description=("V7 split P/I geometry refinement of V6: metadata-based per-zone PI gains. Reference area: 2484 CV; "
                             f"P exponent={params['area_exponent']}, I exponent={params['integral_area_exponent']}. "
                             "Gain scale clipped to [0.5,2]; maximum integral contribution retained. "
                             "AHU/pump coordination, bands and 288-step cooling hold unchanged. "
                             "Lower full training/dev means than V6, V5 and V4; standalone parity checked. No public-score claim.")
                (notebook.parent/"SUBMISSION_DESCRIPTION.txt").write_text(description,encoding="utf-8")
                validation=dict(protocol_sha256=fingerprint,validation_scope="complete local cohorts, not Kaggle private score",
                                training=training,dev=dev,notebook_sha256=digest(notebook))
                write_json(notebook.parent/"validation.json",validation)
                break
            if "submission_notebook" in decision:
                break
        if "submission_notebook" not in decision:
            decision["eligible_for_packaging"]=False
            decision["stop_reason"]="no split rule beat V6, V5 and V4 on both full means; retain existing submissions"
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


def main():
    with RunLock(OUT/"scheduler.lock"):
        run_study()


if __name__ == "__main__":
    main()

