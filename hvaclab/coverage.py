"""Resumable full-episode coverage using isolated interpreters and disk results.

Run from the repository root: python -m hvaclab.coverage --workers 2
No multiprocessing pipes are needed. Failed/short runs never count as complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_checkpoint(temporary, target, attempts=7, initial_delay=.02):
    """Retry transient Windows sharing locks, preserving the old checkpoint."""
    temporary, target = Path(temporary), Path(target)
    for attempt in range(attempts):
        try:
            temporary.replace(target)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt + 1 >= attempts:
                raise
            time.sleep(initial_delay * 2 ** attempt)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    replace_checkpoint(temporary, path)


def valid_result(row, task):
    return (all(row.get(key) == task[key] for key in ("building", "weather", "policy", "steps"))
            and not row.get("error")
            and all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                    for key in ("total_reward", "mean_reward", "prod_regret", "energy_cost", "carbon"))
            and abs(row["mean_reward"] * row["steps"] - row["total_reward"]) < 1e-7)


def source_snapshot(plan):
    # Cache provenance covers the policy files; additional hashes freeze the
    # simulator and runner for every newly launched rollout and resumed run.
    for name, expected in plan["policy_source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"policy differs from planned/cache source: {name}")
    paths = {Path("hvaclab/registry.py"), Path("hvaclab/runner.py"), Path(__file__).relative_to(Path.cwd())}
    paths.update(Path("smart_buildings").rglob("*.py"))
    paths.update(Path("hvaclab/policies").glob("*.py"))
    for split in plan["splits"].values():
        paths.update(Path(split["scenarios_dir"]).rglob("*.gin"))
    return {p.as_posix(): digest(p) for p in sorted(paths)}


def prepare(plan):
    from hvaclab.runner import shipped_pairs
    from hvaclab.registry import _VARIANTS
    if _VARIANTS[plan["policy"]][1] != plan["parameters"]:
        raise ValueError("registry parameters differ from the plan")
    tasks, cached, expected_sets = [], {}, {}
    for split_name, split in plan["splits"].items():
        pairs = set(shipped_pairs(Path(split["scenarios_dir"])))
        if len(pairs) != split["expected_count"]:
            raise ValueError(f"unexpected scenario count: {split_name}")
        expected_sets[split_name] = pairs
        rows = json.loads(Path(split["cache_file"]).read_text())["results"]
        seen = set()
        for row in rows:
            pair = (row["building"], row["weather"])
            task = dict(building=pair[0], weather=pair[1], policy=plan["policy"], steps=plan["steps"])
            if pair not in pairs or pair in seen or not valid_result(row, task):
                raise ValueError(f"invalid/duplicate cache row: {split_name}/{pair}")
            seen.add(pair)
        missing = {(r["building"], r["weather"]) for r in split["missing"]}
        if seen | missing != pairs or seen & missing or len(rows) != split["cached_count"]:
            raise ValueError(f"cache/missing partition mismatch: {split_name}")
        cached[split_name] = rows
        for building, weather in sorted(missing):
            tasks.append(dict(split=split_name, building=building, weather=weather,
                              policy=plan["policy"], steps=plan["steps"],
                              scenarios_dir=split["scenarios_dir"]))
    return tasks, cached, expected_sets


def run_job(path):
    from hvaclab.runner import _run_one
    job = json.loads(Path(path).read_text())
    task = job["task"]
    for name, expected in job["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"source changed before rollout: {name}")
    row = _run_one((task["building"], task["weather"], task["policy"],
                    task["steps"], task["scenarios_dir"]))
    write_json(job["result_path"], dict(task=task, fingerprint=job["fingerprint"], result=row))
    if not valid_result(row, task):
        raise SystemExit(1)


def run(plan_path, output, workers):
    from hvaclab.runner import ScenarioResult, save
    plan = json.loads(Path(plan_path).read_text())
    tasks, cached, expected = prepare(plan)
    snapshot = source_snapshot(plan)
    provenance = dict(plan_sha256=digest(plan_path), source_sha256=snapshot,
                      cache_sha256={s["cache_file"]: digest(s["cache_file"]) for s in plan["splits"].values()})
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    manifest = output / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("coverage sources/cache changed; choose a new output directory")
    write_json(manifest, provenance)
    completed, pending, failed, active = [], [], [], []
    for task in tasks:
        stem = f"{task['split']}__{task['building']}__{task['weather']}"
        result_path = output / "jobs" / (stem + ".result.json")
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if record["task"] != task or record["fingerprint"] != fingerprint:
                raise ValueError(f"stale result: {result_path}")
            if valid_result(record["result"], task):
                completed.append(record)
                continue
        descriptor = output / "jobs" / (stem + ".job.json")
        write_json(descriptor, dict(task=task, result_path=str(result_path),
                                   fingerprint=fingerprint, source_sha256=snapshot))
        pending.append((task, descriptor, result_path))

    def checkpoint(status):
        counts = {}
        for split_name in cached:
            rows = cached[split_name] + [r["result"] for r in completed if r["task"]["split"] == split_name]
            counts[split_name] = dict(completed=len(rows), expected=len(expected[split_name]))
            # Atomic replacement of each aggregate file as well as job files.
            destination = output / f"{split_name}.json"
            temporary = destination.with_suffix(".tmp")
            save([ScenarioResult(**r) for r in rows], temporary)
            temporary.replace(destination)
        write_json(output / "status.json", dict(status=status, fingerprint=fingerprint,
                   counts=counts, new_completed=len(completed), new_total=len(tasks),
                   failed=failed, active=[a[0] for a in active], pending=len(pending),
                   updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))

    checkpoint("running")
    heartbeat = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < workers:
                # Leave room for the desktop and a worker's sparse-factor peak.
                import psutil
                if active and psutil.virtual_memory().available < 1.5 * 2**30:
                    break
                task, descriptor, result_path = pending.pop(0)
                log = descriptor.with_suffix(".log").open("w", encoding="utf-8")
                process = subprocess.Popen([sys.executable, "-m", "hvaclab.coverage", "--job", str(descriptor)],
                                           stdout=log, stderr=subprocess.STDOUT)
                active.append((task, process, log, result_path))
                print(f"START {task['split']} {task['building']}/{task['weather']} pid={process.pid}", flush=True)
                checkpoint("running")
            changed = False
            for item in list(active):
                task, process, log, result_path = item
                code = process.poll()
                if code is None:
                    continue
                active.remove(item)
                log.close()
                record = json.loads(result_path.read_text()) if result_path.exists() else None
                if (code == 0 and record and record["task"] == task
                        and record["fingerprint"] == fingerprint and valid_result(record["result"], task)):
                    completed.append(record)
                    print(f"DONE {task['split']} {task['building']}/{task['weather']} "
                          f"score={abs(record['result']['mean_reward']):.8f} "
                          f"seconds={record['result']['seconds']:.1f}", flush=True)
                else:
                    failed.append(dict(task=task, exit_code=code, result_path=str(result_path)))
                    print(f"FAIL {task['split']} {task['building']}/{task['weather']}", flush=True)
                changed = True
            if changed or time.monotonic() - heartbeat >= 60:
                checkpoint("running")
                heartbeat = time.monotonic()
                print(f"PROGRESS {len(completed)}/{len(tasks)} new complete; "
                      f"{len(active)} running; {len(failed)} failed", flush=True)
            if active:
                time.sleep(2)
        checkpoint("failed" if failed else "complete")
    except BaseException:
        for _, process, log, _ in active:
            process.terminate()
            process.wait()
            log.close()
        active.clear()
        checkpoint("interrupted")
        raise
    if failed:
        raise SystemExit(1)


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/plans/e_sync_coverage.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/e_sync_coverage"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.job:
        run_job(args.job)
    else:
        run(args.plan, args.out, args.workers)


if __name__ == "__main__":
    main()
