"""Common full-cohort runner for the two queued Version 8 follow-ups."""
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

OUT = None
CONTROL_PAIR = ("building_train_1", "fairbanks_1")
CANDIDATES = ()
ALL_CANDIDATES = ()
CONTROL = None
STUDY = None
IMPROVEMENT_EPS = 1e-10



def run_study():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from hvaclab.pipeline_entry import register
    register()
    from hvaclab import runner
    from hvaclab.registry import _VARIANTS
    refs = {}
    v4_refs = {}
    curve_path=Path("experiments/runs/reheat_curve_calibration/rollouts.json")
    curve_rows=[r for r in runner.load(curve_path) if r.policy=="valve8_gentle"]
    if len(curve_rows)!=50 or len({(r.building,r.weather) for r in curve_rows})!=50:
        raise ValueError("need 50 unique submitted-curve reference episodes")
    for row in curve_rows:
        if not valid_result(vars(row),dict(building=row.building,weather=row.weather,
                                         policy="valve8_gentle",steps=8640)):
            raise ValueError("invalid submitted-curve reference episode")
    for split in ("training", "dev"):
        v4_refs[split] = {(r.building, r.weather): r for r in runner.load(
            Path("experiments/runs/e_sync_coverage") / f"{split}.json")}
        if len(v4_refs[split])!=(30 if split=="training" else 20):
            raise ValueError("need complete V4 reference coverage")
        for row in v4_refs[split].values():
            if not valid_result(vars(row),dict(building=row.building,weather=row.weather,
                                              policy="e_sync",steps=8640)):
                raise ValueError("invalid V4 baseline episode")
        refs[split]={(r.building,r.weather):r for r in curve_rows
                     if (r.building,r.weather) in v4_refs[split]}
        if len(refs[split])!=len(v4_refs[split]):
            raise ValueError("curve references do not match V4 scenarios")
    if set(refs["training"]) & set(refs["dev"]):
        raise ValueError("reference cohorts overlap")
    from hvaclab.pipeline_entry import SPECS
    dependencies=('persistence_calibration',)+SPECS[STUDY]['dependencies']
    comparison_groups=[]
    comparison_hashes={}
    for dependency in dependencies:
        preceding=Path("experiments/runs")/dependency
        state_path=preceding/"status.json"
        result_path=preceding/"rollouts.json"
        state=json.loads(state_path.read_text())
        if state['status']!='complete':raise ValueError('dependency not complete: '+dependency)
        comparison_hashes[state_path.as_posix()]=digest(state_path)
        comparison_hashes[result_path.as_posix()]=digest(result_path)
        selected=state['decision'].get('selected')
        if selected:
            winner_rows=[r for r in runner.load(result_path) if r.policy==selected]
            if len(winner_rows)!=50 or len({(r.building,r.weather) for r in winner_rows})!=50:
                raise ValueError('need 50 unique full winner episodes: '+dependency)
            for row in winner_rows:
                if not valid_result(vars(row),dict(building=row.building,weather=row.weather,
                                                  policy=selected,steps=8640)):
                    raise ValueError('invalid winner: '+dependency)
            group={split:{(r.building,r.weather):r for r in winner_rows
                          if (r.building,r.weather) in refs[split]} for split in refs}
            if any(len(group[split])!=len(refs[split]) for split in refs):
                raise ValueError('winner identities mismatch: '+dependency)
            comparison_groups.append((dependency,selected,group))
    base_key=SPECS[STUDY].get('base_key','persist_432')
    base_run=SPECS[STUDY].get('control_run','persistence_calibration')
    fixed_rows=[r for r in runner.load(Path('experiments/runs')/base_run/'rollouts.json')
                if r.policy==base_key]
    if len(fixed_rows)!=50:raise ValueError('control reference missing: '+base_key)
    fixed_refs={split:{(r.building,r.weather):r for r in fixed_rows
                       if (r.building,r.weather) in refs[split]} for split in refs}
    dev_pairs = sorted(refs["dev"])
    train_pairs = sorted(refs["training"])
    plan = json.loads(Path("experiments/plans/e_sync_coverage.json").read_text())
    snapshot = source_snapshot(plan)
    snapshot["hvaclab/pipeline_calibration.py"] = digest(__file__)
    for path in ("hvaclab/submission/experimental.py", "hvaclab/submission/build.py"):
        snapshot[path] = digest(path)
    snapshot["hvaclab/pipeline_entry.py"]=digest("hvaclab/pipeline_entry.py")
    snapshot["hvaclab/pipeline_queue.py"]=digest("hvaclab/pipeline_queue.py")
    snapshot[f"hvaclab/{STUDY}.py"]=digest(f"hvaclab/{STUDY}.py")
    snapshot["hvaclab/pi_calibration.py"]=digest("hvaclab/pi_calibration.py")
    protocol = dict(comparison_sha256=comparison_hashes,curve_reference_sha256=digest(curve_path),source_sha256=snapshot,
                    reference_sha256={split: digest(Path("experiments/runs/e_sync_coverage") / f"{split}.json") for split in refs},
                    variants={p: _VARIANTS[p][1] for p in (CONTROL,)+ALL_CANDIDATES},
                    calibration_batches=[CANDIDATES],
                    full_dev_pairs=dev_pairs, full_training_pairs=train_pairs,
                    control_pair=CONTROL_PAIR, steps=8640, max_workers=3,
                    available_memory_gib_per_launch=1.75, launch_spacing_seconds=20,
                    available_disk_gib_per_launch=1.,
                    improvement_epsilon=IMPROVEMENT_EPS,
                    dependency_versions={p: importlib.metadata.version(p) for p in
                                         ("numpy","scipy","pandas","gin-config","protobuf")},
                    objective=SPECS[STUDY]['title']+". Full dev then ranked full training; "
                              "both means must beat V7, V4 and every selected predecessor winner. "
                              "No scenario labels, geometry, private state or test-split access.")



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
        v4=sum(v4_refs[split][p].score for p in pairs)/len(pairs)
        winner_scores={name+':'+key:sum(group[split][p].score for p in pairs)/len(pairs)
                       for name,key,group in comparison_groups}
        winner_score=min(winner_scores.values(),default=baseline)
        worst_ratio = max(abs(r["mean_reward"])/refs[split][(r["building"],r["weather"])].score for r in rows)
        comparisons = [dict(scenario=r["building"]+"/"+r["weather"],
                            score=abs(r["mean_reward"]),
                            submitted_curve=refs[split][(r["building"],r["weather"])].score,
                            difference=abs(r["mean_reward"])-refs[split][(r["building"],r["weather"])].score)
                       for r in rows]
        return dict(n=len(rows), score=score, submitted_curve=baseline,winner_scores=winner_scores,winner_score=winner_score, e_sync=v4,
                    change_vs_v4_pct=100*(score/v4-1),
                    change_pct=100*(score/baseline-1), worst_ratio=worst_ratio,
                    improved=sum(c["difference"] < -IMPROVEMENT_EPS for c in comparisons),
                    regressed=sum(c["difference"] > IMPROVEMENT_EPS for c in comparisons),
                    largest_increases=sorted(comparisons,key=lambda c:c["difference"],reverse=True)[:3],
                    comfort=-.6*sum(r["prod_regret"]/r["steps"] for r in rows)/len(rows),
                    energy=.2*sum(r["energy_cost"]/r["steps"] for r in rows)/len(rows),
                    carbon=.2*sum(r["carbon"]/r["steps"] for r in rows)/len(rows),
                    passes=score < min(baseline,v4,winner_score)-IMPROVEMENT_EPS)

    def checkpoint(status="running"):
        runner.save([runner.ScenarioResult(**r["result"]) for r in records.values()], OUT / "rollouts.json")
        state = dict(status=status, stage=stage, completed=len(records), failed=failed,
                     waiting_for_memory=memory_waiting, waiting_for_disk=disk_waiting,
                     active=[item[0] for item in active], decision=decision,
                     counts={p: {split: sum(1 for k in records if k[0]==split and k[1]==p)
                                 for split in ("dev","training")} for p in ALL_CANDIDATES},
                     updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(OUT / "status.json", state)
        lines = ["# Queued Version 8 follow-up", "", f"Status: {status}; stage: {stage}.", "",
                 "All scores use complete 8,640-step episodes against matched submitted-curve references; V4 is checked separately.", "",
                 "| Cohort | Variant | n | Submitted curve | Candidate | Change |", "|---|---|---:|---:|---:|---:|"]
        tex = ["\\paragraph{Automatically recorded closed-loop results.}",
               "Status: \\texttt{" + status.replace("_", "\\_") + "}; stage: \\texttt{" + stage.replace("_", "\\_") + "}.",
               "All entries below require complete 8,640-step episodes on matching scenarios.",
               "\\begin{center}\\small", "\\begin{tabular}{llrrrr}\\toprule",
               "Cohort & Variant & $n$ & Curve & Candidate & Change \\\\\\midrule"]
        cohorts = [("full dev", "dev", dev_pairs), ("full training", "training", train_pairs)]
        for label, split, pairs in cohorts:
            for policy in ALL_CANDIDATES:
                s = statistics(split, policy, pairs)
                if s is None:
                    continue
                lines.append(f"| {label} | {policy} | {s['n']} | {s['submitted_curve']:.8f} | {s['score']:.8f} | {s['change_pct']:+.2f}% |")
                tex.append(f"{label} & \\texttt{{{policy.replace('_', chr(92)+'_')}}} & {s['n']} & "
                           f"{s['submitted_curve']:.8f} & {s['score']:.8f} & {s['change_pct']:+.2f}\\% \\\\")
        tex += ["\\bottomrule\\end{tabular}\\end{center}",
                "These are complete matched cohorts. Dev is repeatedly used model-selection data, not untouched validation. Individual regressions are reported without a hard cutoff.",
                "Completed new rollouts: " + str(len(records)) + ". No candidate replaces Version 4 automatically."]
        if "unchanged_curve_full_episode_parity" in decision:
            tex.append("The fresh baseline full-episode reproduction gate "
                       +( "passed." if decision["unchanged_curve_full_episode_parity"] else "failed."))
        if "stop_reason" in decision:
            tex.append("Experiment decision: " + decision["stop_reason"] + ".")
        if "selected" in decision:
            tex.append("The selected candidate is \\texttt{" + decision["selected"].replace("_", "\\_") + "}.")
        lines += ["", "Decision:", "", "```json", json.dumps(decision, indent=2), "```", "",
                  "Earlier paper sections and artifacts remain unchanged. Percentages compare with the submitted curve; JSON also reports V4. "
                  "Selection requires full-cohort means below BOTH V4 and the submitted curve. Full-training failures remain recorded while the next dev-ranked candidate is tried."]
        (Path(f"reports/{STUDY}_REVIEW.md")).write_text("\n".join(lines), encoding="utf-8")
        (Path(f"reports/paper/{STUDY}_results.tex")).write_text("\n".join(tex)+"\n", encoding="utf-8")

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
                process = subprocess.Popen([sys.executable, "-m", "hvaclab.pipeline_entry", "--job", str(descriptor)],
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
        execute([make_task("training",CONTROL,CONTROL_PAIR)])
        control=records[("training",CONTROL,*CONTROL_PAIR)]["result"]
        parity=abs(control["mean_reward"]-fixed_refs["training"][CONTROL_PAIR].mean_reward)<=1e-10
        decision["unchanged_curve_full_episode_parity"]=parity
        if not parity:raise RuntimeError("full control mismatch: "+base_key)
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
                    subprocess.run([sys.executable,"-u","-m","hvaclab.pipeline_entry","--package",
                                    "--variant",selected,"--verify"],stdout=log,stderr=subprocess.STDOUT,check=True)
                notebook=Path("experiments/submission")/selected/"submission.ipynb"
                decision["submission_notebook"]=notebook.as_posix()
                params=_VARIANTS[selected][1]
                description=(f"{STUDY}: variant={selected}; hold={params['hold_steps']} steps "
                             f"({params['hold_steps']/12:g}h). "
                             f"Reheat exponent={params['reheat_exponent']}; "
                             f"heating damper exponent={params.get('heating_damper_exponent',1.0)}. "
                             "Lower full local means than V7, V4 and selected predecessor winners. "
                             "Standalone parity verified; runtime source marker printed. No organizer claim.")
                (notebook.parent/"SUBMISSION_DESCRIPTION.txt").write_text(description,encoding="utf-8")
                validation=dict(protocol_sha256=fingerprint,validation_scope="complete local cohorts, not Kaggle private score",
                                training=training,dev=dev,notebook_sha256=digest(notebook))
                write_json(notebook.parent/"validation.json",validation)
                break
            if "submission_notebook" in decision:
                break
        if "submission_notebook" not in decision:
            decision["eligible_for_packaging"]=False
            decision["stop_reason"]="no candidate beat all reference means on both cohorts; retain existing submissions"
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


def main(study):
    global OUT,CONTROL,CANDIDATES,ALL_CANDIDATES,STUDY
    from hvaclab.pipeline_entry import SPECS
    spec=SPECS[study]
    STUDY=study
    OUT=Path('experiments/runs')/study
    CONTROL=spec['control']
    CANDIDATES=tuple(key for key in spec['arms'] if key!=CONTROL)
    ALL_CANDIDATES=CANDIDATES
    with RunLock(OUT/'scheduler.lock'):
        run_study()
