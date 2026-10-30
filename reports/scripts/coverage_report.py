"""Report complete, matched coverage; optionally wait and run shadow diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def read_rows(path, policy):
    rows = json.loads(Path(path).read_text())["results"]
    indexed = {}
    for row in rows:
        if row["policy"] != policy:
            continue
        pair = row["building"], row["weather"]
        if pair in indexed or row.get("error") or row["steps"] != 8640:
            raise ValueError(f"duplicate, failed or short row in {path}: {pair}")
        indexed[pair] = row
    return indexed


def compare(candidate, reference):
    if not candidate or set(candidate) != set(reference):
        raise ValueError("full comparison requires identical, nonempty scenario sets")
    def terms(rows):
        n = len(rows)
        return dict(score=sum(abs(r["mean_reward"]) for r in rows.values()) / n,
                    comfort=-.6 * sum(r["prod_regret"] / r["steps"] for r in rows.values()) / n,
                    energy=.2 * sum(r["energy_cost"] / r["steps"] for r in rows.values()) / n,
                    carbon=.2 * sum(r["carbon"] / r["steps"] for r in rows.values()) / n)
    trial, base = terms(candidate), terms(reference)
    pairs = []
    for pair in sorted(candidate):
        a, b = abs(candidate[pair]["mean_reward"]), abs(reference[pair]["mean_reward"])
        pairs.append(dict(scenario="/".join(pair), e_sync=a, h_288=b, difference=a-b,
                          change_pct=100*(a/b-1) if b else None))
    return dict(n=len(candidate), e_sync=trial, h_288=base,
                change_pct=100*(trial["score"]/base["score"]-1),
                improved=sum(p["difference"] < -1e-10 for p in pairs),
                regressed=sum(p["difference"] > 1e-10 for p in pairs),
                scenarios=pairs)


def report(folder):
    folder = Path(folder)
    status = json.loads((folder / "status.json").read_text())
    comparisons = {}
    for split, reference, count in (("training", "h288_full30.json", 30),
                                     ("dev", "h288_dev.json", 20)):
        trial_path = folder / f"{split}.json"
        trial = read_rows(trial_path, "e_sync")
        base = read_rows(ROOT / "experiments/runs" / reference, "h_288")
        if len(trial) == count:
            comparisons[split] = compare(trial, base)
    lines = ["# e_sync broad coverage", "",
             f"Run status: **{status['status']}**. Updated: {status['updated_at']}.", "",
             "The submitted Version 4 notebook is unchanged. These are local full-episode "
             "measurements, not a Kaggle result. The test split is not used.", "",
             "| Split | Completed | Expected |", "|---|---:|---:|"]
    for split, counts in status["counts"].items():
        lines.append(f"| {split} | {counts['completed']} | {counts['expected']} |")
    lines += ["", "Aggregates in the run directory may be partial. A full-set comparison "
              "appears only after every expected scenario in that split finishes.", ""]
    for split, result in comparisons.items():
        lines += [f"## {split}: {result['n']} matched scenarios", "",
                  f"Score: h_288 **{result['h_288']['score']:.8f}** → "
                  f"e_sync **{result['e_sync']['score']:.8f}** "
                  f"({result['change_pct']:+.2f}%; lower is better). "
                  f"Improved {result['improved']}, regressed {result['regressed']}.", "",
                  "| Policy | Comfort | Energy | Carbon |", "|---|---:|---:|---:|"]
        for name in ("h_288", "e_sync"):
            item = result[name]
            lines.append(f"| {name} | {item['comfort']:.8f} | {item['energy']:.8f} | {item['carbon']:.8f} |")
        lines += ["", "Largest increases (negative values are improvements):", "",
                  "| Scenario | h_288 | e_sync | Difference |", "|---|---:|---:|---:|"]
        for item in sorted(result["scenarios"], key=lambda p: p["difference"], reverse=True)[:10]:
            lines.append(f"| {item['scenario']} | {item['h_288']:.8f} | {item['e_sync']:.8f} | {item['difference']:+.8f} |")
        lines.append("")
    lines += ["## Reproducibility", "",
              "Resume with `python -m hvaclab.coverage --workers 2` from the repository root "
              "only after checking that the existing coverage process has stopped. "
              "Do not launch two schedulers for the same output directory.", "",
              "Each job has an isolated interpreter, log, descriptor and atomically saved result. "
              "The provenance file freezes policy/simulator/runner sources, scenario gin files, "
              "and the two explicitly reused caches. Failed or short runs cannot complete coverage.", "",
              "Source/config/cache hashes protect against code drift; external weather and geometry "
              "data files are not hashed by this runner and must remain unchanged during the sweep.", ""]
    destination = ROOT / "reports/E_SYNC_COVERAGE.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    payload = dict(status=status, comparisons=comparisons)
    (folder / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {status['status']}; complete splits={list(comparisons)}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "experiments/runs/e_sync_coverage")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--shadow-after", action="store_true")
    args = parser.parse_args()
    while True:
        status = report(args.folder)
        if not args.watch or status["status"] != "running":
            break
        time.sleep(30)
    if args.shadow_after and status["status"] == "complete":
        # This runs after the sweep, preserving its two-process memory budget.
        with (args.folder / "shadow.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", "-m", "hvaclab.delay_shadow"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Shadow trace and forecast metrics complete.", flush=True)


if __name__ == "__main__":
    main()
