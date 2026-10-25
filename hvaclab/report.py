"""Compare a sweep against the shipped baselines, per scenario.

The leaderboard is a mean over 30 equally weighted scenarios, and the worst
few own most of it, so a single aggregate number hides the thing worth looking
at. This prints the per-scenario table with the baselines beside it and sorts
by where the loss actually is.

    python -m hvaclab.report experiments/runs/occ.json
    python -m hvaclab.report experiments/runs/occ.json --against fixed_setpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hvaclab import runner

BASELINE_RETURNS = Path("baseline_returns.json")
FULL_EPISODE_STEPS = runner.FULL_EPISODE_STEPS
BASELINES = ("bang_bang", "fixed_setpoint", "pid", "pid_water")


def baseline_scores() -> dict[str, dict[str, float]]:
    """Per-scenario leaderboard contribution of each shipped baseline.

    `baseline_returns.json` holds total reward over an 8,640-step episode, and
    the metric is the absolute per-step mean, so this is just |total| / 8640.
    """
    raw = json.loads(BASELINE_RETURNS.read_text())
    return {
        scenario: {
            policy: abs(total) / FULL_EPISODE_STEPS
            for policy, total in policies.items()
        }
        for scenario, policies in raw.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path, help="a sweep JSON from hvaclab.sweep")
    parser.add_argument("--against", default="pid_water",
                        help="baseline to difference against (default pid_water)")
    args = parser.parse_args()

    results = runner.load(args.sweep)
    base = baseline_scores()
    ref = args.against

    by_policy: dict[str, list] = {}
    for r in results:
        by_policy.setdefault(r.policy, []).append(r)

    for policy, rows in by_policy.items():
        rows = [r for r in rows if not r.error]
        if not rows:
            print(f"\n{policy}: every run failed")
            continue

        # Only scenarios with a reference entry can be differenced; generated
        # holdout scenarios have none, and are reported without one.
        have_ref = [r for r in rows if r.scenario in base]

        print(f"\n{'=' * 104}")
        print(f"{policy}   ({len(rows)} scenarios, "
              f"{rows[0].steps} steps)")
        print("=" * 104)
        print(f"{'scenario':32s}{'score':>9s}{'  vs ' + ref:>14s}"
              f"{'delta':>9s}{'comfort':>9s}{'energy':>8s}{'carbon':>8s}"
              f"{'empty%':>8s}")
        print("-" * 104)

        # Worst first: that is the order the metric actually cares about.
        for r in sorted(rows, key=lambda r: -r.score):
            n = max(r.steps, 1)
            comfort = -0.6 * r.prod_regret / n
            energy = 0.2 * r.energy_cost / n
            carbon = 0.2 * r.carbon / n
            empty_share = (
                100.0 * (-r.reward_unoccupied) / (-r.total_reward)
                if r.total_reward < 0 else 0.0
            )
            if r.scenario in base and ref in base[r.scenario]:
                b = base[r.scenario][ref]
                delta = r.score - b
                mark = "+" if delta > 0 else ""
                ref_s, delta_s = f"{b:.5f}", f"{mark}{delta:.5f}"
            else:
                ref_s, delta_s = "-", "-"
            print(f"{r.scenario:32s}{r.score:9.5f}{ref_s:>14s}{delta_s:>9s}"
                  f"{comfort:9.5f}{energy:8.5f}{carbon:8.5f}{empty_share:7.1f}%")

        print("-" * 104)
        score = sum(r.score for r in rows) / len(rows)
        print(f"{'MEAN (leaderboard metric)':32s}{score:9.5f}")

        if have_ref:
            ref_mean = sum(base[r.scenario][ref] for r in have_ref) / len(have_ref)
            own_mean = sum(r.score for r in have_ref) / len(have_ref)
            better = sum(
                1 for r in have_ref if r.score < base[r.scenario][ref]
            )
            print(f"{'  on the ' + str(len(have_ref)) + ' shared scenarios':32s}"
                  f"{own_mean:9.5f}{ref_mean:14.5f}"
                  f"{own_mean - ref_mean:+9.5f}")
            print(f"  better than {ref} on {better}/{len(have_ref)} scenarios")

        if len(have_ref) == len(rows) and rows[0].steps == FULL_EPISODE_STEPS:
            print("\n  all shipped baselines, for reference:")
            for b in BASELINES:
                vals = [base[r.scenario][b] for r in have_ref
                        if b in base[r.scenario]]
                if vals:
                    print(f"    {b:16s}{sum(vals) / len(vals):9.5f}")


if __name__ == "__main__":
    main()
