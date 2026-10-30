"""Gated pump-pressure experiment with memory-aware scheduling and packaging."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from hvaclab.coverage import digest, source_snapshot, valid_result, write_json

OUT = Path("experiments/runs/prestart")
TRAIN = [("building_train_1", "fairbanks_1"), ("building_train_7", "phoenix_3"),
         ("building_train_4", "atlanta_5"), ("building_train_7", "denver_5")]
DEV = [("building_9002", "phoenix_2"), ("building_9002", "atlanta_4"),
       ("building_9002", "singapore_2"), ("building_9001", "fairbanks_3")]


def main():
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
    plan = json.loads(Path("experiments/plans/e_sync_coverage.json").read_text())
    snapshot = source_snapshot(plan)
    snapshot["hvaclab/prestart_experiment.py"] = digest(__file__)
    for path in ("hvaclab/submission/experimental.py", "hvaclab/submission/build.py"):
        snapshot[path] = digest(path)
    profile_path = Path("experiments/runs/prestart_profile_policyclock.json")
    if not profile_path.exists():
        raise ValueError("complete the morning_profile diagnostic first")
    diagnostic = json.loads(profile_path.read_text())
    for path, expected in diagnostic["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"diagnostic source changed: {path}")
    protocol = dict(diagnostic_sha256=digest(profile_path),
                    diagnostic_minimum_first_two_hours_comfort_share=0.25,
                    source_sha256=snapshot,
                    reference_sha256={s: digest(Path("experiments/runs/e_sync_coverage") / f"{s}.json") for s in refs},
                    variants={p: _VARIANTS[p][1] for p in ("a_start_ctrl", "a_start2", "a_start3")},
                    training_screen=TRAIN, dev_screen=DEV, steps=8640,
                    gate="zero parity <=1e-10; mean score lower and no scenario >10% worse; "
                         "all training-passing candidates get dev screen; select lowest passing dev mean; "
                         "only passing candidate advances to full training/dev coverage")
    # JSON canonicalization also normalizes tuples for later resume comparison.
    protocol = json.loads(json.dumps(protocol))
    provenance = OUT / "protocol.json"
    if provenance.exists() and json.loads(provenance.read_text()) != protocol:
        raise ValueError("experiment sources or protocol changed; preserve results and use a new directory")
    write_json(provenance, protocol)
    fingerprint = digest(provenance)
    records, failed, active = {}, [], []
    memory_waiting = False
    stage = "training_screen"
    decision = dict(first_two_hours_comfort_share=diagnostic["first_two_hours_comfort_share"],
                    diagnostic_supports_morning=diagnostic["first_two_hours_comfort_share"] >= .25)

    def key(task):
        return task["split"], task["policy"], task["building"], task["weather"]

    def make_task(split, policy, pair):
        return dict(split=split, policy=policy, building=pair[0], weather=pair[1], steps=8640,
                    scenarios_dir="smart_buildings/training_scenarios" if split == "training"
                    else "smart_buildings/holdout_scenarios/dev")

    control_task = make_task("training", "a_start_ctrl", TRAIN[1])
    if not valid_result(diagnostic["result"], control_task):
        raise ValueError("diagnostic is not the expected complete control episode")
    if abs(diagnostic["result"]["mean_reward"] - refs["training"][TRAIN[1]].mean_reward) > 1e-10:
        raise ValueError("diagnostic control does not reproduce cached e_sync")
    control_path = OUT / "jobs" / ("__".join(key(control_task)) + ".result.json")
    if not control_path.exists():
        write_json(control_path, dict(task=control_task, result=diagnostic["result"],
                                     fingerprint=fingerprint, reused_from=str(profile_path),
                                     diagnostic_sha256=digest(profile_path)))
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
        return dict(n=len(rows), score=score, e_sync=baseline,
                    change_pct=100*(score/baseline-1), worst_ratio=worst_ratio,
                    passes=score < baseline and worst_ratio <= 1.10)

    def checkpoint(status="running"):
        runner.save([runner.ScenarioResult(**r["result"]) for r in records.values()], OUT / "rollouts.json")
        state = dict(status=status, stage=stage, completed=len(records), failed=failed,
                     waiting_for_memory=memory_waiting,
                     active=[item[0] for item in active], decision=decision,
                     updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json(OUT / "status.json", state)
        lines = ["# Earlier PI preconditioning", "", f"Status: {status}; stage: {stage}.", "",
                 "All scores use complete 8,640-step episodes against matched e_sync references.",
                 "The diagnostic episode is explicitly reused as the control, not counted twice as independent evidence.", "",
                 "| Cohort | Variant | n | e_sync | Candidate | Change |", "|---|---|---:|---:|---:|---:|"]
        tex = ["\\paragraph{Automatically recorded closed-loop results.}",
               "Status: \\texttt{" + status.replace("_", "\\_") + "}; stage: \\texttt{" + stage.replace("_", "\\_") + "}.",
               "All entries below require complete 8,640-step episodes on matching scenarios.",
               "\\begin{center}\\small", "\\begin{tabular}{llrrrr}\\toprule",
               "Cohort & Variant & $n$ & e\\_sync & Candidate & Change \\\\\\midrule"]
        cohorts = [("train screen", "training", TRAIN), ("dev screen", "dev", DEV),
                   ("full training", "training", sorted(refs["training"])),
                   ("full dev", "dev", sorted(refs["dev"]))]
        for label, split, pairs in cohorts:
            for policy in ("a_start2", "a_start3"):
                s = statistics(split, policy, pairs)
                if s is None:
                    continue
                lines.append(f"| {label} | {policy} | {s['n']} | {s['e_sync']:.8f} | {s['score']:.8f} | {s['change_pct']:+.2f}% |")
                tex.append(f"{label} & \\texttt{{{policy.replace('_', chr(92)+'_')}}} & {s['n']} & "
                           f"{s['e_sync']:.8f} & {s['score']:.8f} & {s['change_pct']:+.2f}\\% \\\\")
        tex += ["\\bottomrule\\end{tabular}\\end{center}",
                "The training and dev screens are subsets of their respective full cohorts, not independent repetitions.",
                "Completed recorded rollouts: " + str(len(records)) + ". No candidate replaces Version 4 automatically.",
                "The hourly diagnostic is reused as the control, not an additional independent repetition.",
                f"First two scheduled hours account for {100*diagnostic['first_two_hours_comfort_share']:.2f}\\% of the diagnostic comfort loss."]
        if "unchanged_prestart_full_episode_parity" in decision:
            control = records[("training", "a_start_ctrl", *TRAIN[1])]["result"]
            tex.append("The unchanged-pump full-episode control score is "
                       f"${abs(control['mean_reward']):.8f}$; exact-reproduction gate: "
                       + ("passed." if decision["unchanged_prestart_full_episode_parity"] else "failed."))
        if "stop_reason" in decision:
            tex.append("Experiment decision: " + decision["stop_reason"] + ".")
        if "selected" in decision:
            tex.append("The selected candidate is \\texttt{" + decision["selected"].replace("_", "\\_") + "}.")
        lines += ["", "Decision:", "", "```json", json.dumps(decision, indent=2), "```", "",
                  "Earlier paper sections and the e_sync public score remain unchanged. "
                  "Screens are subsets of full cohorts, not additional independent evidence."]
        (Path("reports/PRESTART_REVIEW.md")).write_text("\n".join(lines), encoding="utf-8")
        (Path("reports/paper/prestart_results.tex")).write_text("\n".join(tex)+"\n", encoding="utf-8")

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
            while pending and len(active) < 2:
                import psutil
                # Allow the new interpreter to allocate before checking RAM
                # for another one; avoid two simultaneous initialization peaks.
                if active and time.monotonic() < next_launch:
                    break
                required_gib = 1.5 if active else 1.25
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
    if not decision["diagnostic_supports_morning"]:
        stage = "hypothesis_not_supported"
        decision["stop_reason"] = "first-two-hour comfort share below 25%; do not run the proposed prestart sweep"
        checkpoint("complete")
        compile_paper()
        return
    try:
        execute([make_task("training", "a_start_ctrl", TRAIN[1])]
                + [make_task("training", p, pair) for p in ("a_start2", "a_start3") for pair in TRAIN])
        control = records[("training", "a_start_ctrl", *TRAIN[1])]["result"]
        parity = abs(control["mean_reward"] - refs["training"][TRAIN[1]].mean_reward) <= 1e-10
        decision["unchanged_prestart_full_episode_parity"] = parity
        options = {p: statistics("training", p, TRAIN) for p in ("a_start2", "a_start3")}
        decision["training_screen"] = options
        passing = [p for p,s in options.items() if s["passes"] and parity]
        if passing:
            decision["dev_screens"] = {}
            for policy in passing:
                stage = "dev_screen"
                decision["screening"] = policy
                checkpoint()
                compile_paper()
                execute([make_task("dev", policy, pair) for pair in DEV])
                decision["dev_screens"][policy] = statistics("dev", policy, DEV)
            eligible = [p for p in passing if decision["dev_screens"][p]["passes"]]
            if eligible:
                selected = min(eligible, key=lambda p: decision["dev_screens"][p]["score"])
                decision["selected"] = selected
                stage = "full_coverage"
                checkpoint()
                compile_paper()
                execute([make_task(split, selected, pair) for split in ("training", "dev") for pair in sorted(refs[split])])
                decision["full_training"] = statistics("training", selected, sorted(refs["training"]))
                decision["full_dev"] = statistics("dev", selected, sorted(refs["dev"]))
                ready = decision["full_training"]["passes"] and decision["full_dev"]["passes"]
                decision["eligible_for_packaging"] = ready
                if ready:
                    stage = "packaging"
                    checkpoint()
                    with (OUT / "packaging.log").open("w", encoding="utf-8") as log:
                        subprocess.run([sys.executable, "-u", "-m", "hvaclab.submission.experimental",
                                        "--variant", selected, "--verify"], stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    notebook = Path("experiments/submission") / selected / "submission.ipynb"
                    decision["submission_notebook"] = notebook.as_posix()
                    description = ("e_sync + earlier zone-PI preconditioning. Prestart: "
                                   + str(_VARIANTS[selected][1]["prestart_hours"])
                                   + " hours. Flat bands, PI gains, pump requests, heating-water "
                                   "coordination and 288-step cooling hold unchanged. "
                                   "Validated on complete matched training/dev cohorts; "
                                   "standalone action parity checked. No public-score claim.")
                    (notebook.parent / "SUBMISSION_DESCRIPTION.txt").write_text(description, encoding="utf-8")
                else:
                    decision["stop_reason"] = "full coverage gate failed; keep Version 4"
            else:
                decision["stop_reason"] = "all candidate dev gates failed; no submission change"
        else:
            decision["stop_reason"] = "training or parity gate failed; no dev tuning or submission change"
        stage = "finished"
        checkpoint("complete")
        compile_paper()
    except Exception:
        checkpoint("failed")
        compile_paper()
        raise


if __name__ == "__main__":
    main()
