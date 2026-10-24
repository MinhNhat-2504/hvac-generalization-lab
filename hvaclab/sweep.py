"""Command line entry point for a parallel sweep.

    python -m hvaclab.sweep --policies occ
    python -m hvaclab.sweep --policies occ pid_water --steps 2880 --out runs/x.json

Defaults to the full 8,640-step episode, because that is the only horizon whose
returns are comparable to ``baseline_returns.json`` and to the leaderboard. A
shorter horizon flatters schedule-aware policies -- this one included, since
most of its advantage is in the eco-setback hours a short run truncates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hvaclab import runner
from hvaclab.registry import available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+", required=True,
                        help=f"policy keys; available: {', '.join(available())}")
    parser.add_argument("--steps", type=int, default=runner.FULL_EPISODE_STEPS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--buildings", nargs="*", default=None,
                        help="restrict to these building names")
    parser.add_argument("--weathers", nargs="*", default=None,
                        help="restrict to these weather strip names")
    parser.add_argument("--scenarios-dir", type=Path,
                        default=runner.SCENARIOS_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the sweep to this JSON file")
    args = parser.parse_args()

    pairs = runner.shipped_pairs(args.scenarios_dir)
    if args.buildings:
        pairs = [p for p in pairs if p[0] in set(args.buildings)]
    if args.weathers:
        pairs = [p for p in pairs if p[1] in set(args.weathers)]
    if not pairs:
        raise SystemExit("no scenarios matched")

    print(f"{len(args.policies)} policies x {len(pairs)} scenarios "
          f"x {args.steps} steps")

    results = runner.run_sweep(
        policies=args.policies,
        pairs=pairs,
        num_steps=args.steps,
        workers=args.workers,
        scenarios_dir=args.scenarios_dir,
    )
    runner.print_summary(results)

    failures = [r for r in results if r.error]
    if failures:
        print(f"\n{len(failures)} runs FAILED:")
        for r in failures:
            print(f"  {r.policy} {r.scenario}")

    if args.out:
        runner.save(results, args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
