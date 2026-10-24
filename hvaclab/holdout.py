"""Build held-out scenarios, so generalization is measured rather than assumed.

The leaderboard runs on six buildings and five weather strips that are not in
this repository, drawn from the same generator and the same recorded stations
as the shipped set. Tuning against the 30 shipped scenarios and reading the
result as progress is therefore the central methodological trap in this
competition: the shipped set is a *training* set, and a controller with enough
knobs will fit it.

This module builds two independent replacements for the leaderboard, both from
material the shipped set never uses:

  dev   New buildings crossed with weather strips the shipped set left unused,
        at stations the shipped set does use. Safe to tune against.

  test  New buildings crossed with `minneapolis`, a station that ships as
        generator input but which **no shipped scenario uses at all**. So the
        policy has seen neither the building nor the climate. Read rarely --
        every look at it spends a little of its value as an unbiased estimate.

Of the 48 shipped weather strips, 30 are spoken for by `training_set.yaml`,
leaving 18 free: two each at atlanta, denver, fairbanks, phoenix and singapore,
and all eight at minneapolis.

Buildings are generated at seeds far from the shipped set's (which uses 0-8)
and kept to the zone count the competition states for the evaluation set,
28-85 zones over 2-4 floors. The scenario generator raises `num_ahus` to
`ceil(zones / 10)`, so the footprint decides the air-handler count and a
building generated at the default footprint range can carry more handlers than
any shipped one -- and cost proportionally more to simulate.

Output goes under `smart_buildings/holdout_scenarios/`, which the repository's
`.gitignore` excludes via `**/holdout*/`: held-out data is reproducible from a
seed and must not become something that gets tuned against by being checked in.

    python -m hvaclab.holdout --count 6
    python -m hvaclab.holdout --count 6 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HOLDOUT_DIR = Path("smart_buildings/holdout_scenarios")
BUILDINGS_DIR = Path("smart_buildings/buildings")
WEATHER_DIR = Path("smart_buildings/weather")
TRAINING_SET = Path("smart_buildings/training_set.yaml")

# Seeds for held-out buildings. The shipped six use 0-8 and the starter kit's
# own examples suggest 500+, so these sit clear of both.
DEV_SEED_START = 9000
TEST_SEED_START = 9500

# The station no shipped scenario uses. Reserved for `test` so that set varies
# both building and climate at once.
UNSEEN_STATION = "minneapolis"

# Competition-stated range for the evaluation buildings.
MIN_ZONES, MAX_ZONES = 28, 85


def used_weather_strips() -> set[str]:
    """Strips the shipped 30 scenarios consume."""
    spec = yaml.safe_load(TRAINING_SET.read_text())
    return {s["weather"] for s in spec["scenarios"]}


def free_weather_strips() -> dict[str, list[str]]:
    """Unused strips, grouped by station."""
    used = used_weather_strips()
    groups: dict[str, list[str]] = {}
    for path in sorted(WEATHER_DIR.glob("*.yaml")):
        name = path.stem
        if name in used:
            continue
        station = name.rsplit("_", 1)[0]
        groups.setdefault(station, []).append(name)
    for station in groups:
        groups[station].sort(key=lambda n: int(n.rsplit("_", 1)[1]))
    return groups


def _run(args: list[str], check: bool = True) -> bool:
    """Run a generator subprocess. Returns whether it succeeded."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0 and check:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
        print(f"    failed: {tail[0] if tail else proc.returncode}", flush=True)
    return proc.returncode == 0


def generate_buildings(seeds: list[int]) -> list[str]:
    """Sample building configs at the given seeds and return their stems."""
    _run([
        sys.executable, "-m", "smart_buildings.run_floor_generator",
        "--seeds", *[str(s) for s in seeds],
        "--no-pdf",
        "--output-dir", str(BUILDINGS_DIR),
    ])
    return [f"building_{s:04d}" for s in seeds]


def build_valid_set(
    seed_start: int, count: int, weathers: list[str], output_dir: Path,
    max_attempts: int = 40,
) -> list[str]:
    """Find ``count`` seeds that generate cleanly, and build their scenarios.

    Not every seed yields a usable building. The scenario generator rejects a
    floor plan containing a zone below 20 m2 -- "a floor plan artifact, not a
    room", which diverges in simulation -- and tells the caller to change the
    seed rather than relax the limit. Since the generator is what produced the
    evaluation buildings too, a rejected seed is not something to work around;
    it is a seed the competition would not have used either.

    So seeds are tried one at a time and the failures are skipped, which also
    keeps one bad seed from taking the whole batch down with it.
    """
    accepted: list[str] = []
    seed = seed_start
    attempts = 0

    while len(accepted) < count and attempts < max_attempts:
        attempts += 1
        name = f"building_{seed:04d}"
        if not _run([
            sys.executable, "-m", "smart_buildings.run_floor_generator",
            "--seeds", str(seed), "--no-pdf",
            "--output-dir", str(BUILDINGS_DIR),
        ]):
            print(f"  seed {seed}: floor generation failed, skipping")
            seed += 1
            continue

        if _run([
            sys.executable, "-m", "smart_buildings.run_scenario_generator",
            "--buildings", name,
            "--weathers", *weathers,
            "--output-dir", str(output_dir),
        ]):
            accepted.append(name)
            print(f"  seed {seed}: ok -> {name}")
        else:
            print(f"  seed {seed}: rejected by the scenario generator, skipping")

        seed += 1

    if len(accepted) < count:
        print(f"  WARNING: only {len(accepted)}/{count} seeds usable "
              f"after {attempts} attempts")
    return accepted


def describe(output_dir: Path) -> list[dict]:
    """Zone and air-handler counts for everything built under ``output_dir``.

    Read from each scenario's own ``config.yaml`` rather than recomputed, so
    the numbers are the ones the simulator will actually use.
    """
    rows = []
    for building_dir in sorted(Path(output_dir).iterdir()):
        if not building_dir.is_dir():
            continue
        for scenario_dir in sorted(building_dir.iterdir()):
            cfg = scenario_dir / "config.yaml"
            if not cfg.exists():
                continue
            spec = yaml.safe_load(cfg.read_text()) or {}
            rows.append({
                "building": building_dir.name,
                "weather": scenario_dir.name,
                "zones": spec.get("num_zones"),
                "ahus": spec.get("num_ahus"),
                "floors": (spec.get("floor_plan") or {}).get("num_floors"),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4,
                        help="buildings per split (default 4)")
    parser.add_argument("--dev-strips", type=int, default=1,
                        help="free strips per seen station to use for dev")
    parser.add_argument("--test-strips", type=int, default=2,
                        help="minneapolis strips to use for test")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit")
    args = parser.parse_args()

    free = free_weather_strips()
    print("free weather strips by station:")
    for station, strips in free.items():
        print(f"  {station:12s} {strips}")

    dev_weathers = [
        s for station, strips in free.items() if station != UNSEEN_STATION
        for s in strips[: args.dev_strips]
    ]
    test_weathers = free.get(UNSEEN_STATION, [])[: args.test_strips]

    dev_seeds = list(range(DEV_SEED_START, DEV_SEED_START + args.count))
    test_seeds = list(range(TEST_SEED_START, TEST_SEED_START + args.count))

    print(f"\ndev : {args.count} buildings x {len(dev_weathers)} strips "
          f"= {args.count * len(dev_weathers)} scenarios")
    print(f"      seeds {dev_seeds}")
    print(f"      {dev_weathers}")
    print(f"test: {args.count} buildings x {len(test_weathers)} strips "
          f"= {args.count * len(test_weathers)} scenarios  [{UNSEEN_STATION}]")
    print(f"      seeds {test_seeds}")
    print(f"      {test_weathers}")

    if args.dry_run:
        return

    for split, seeds, weathers in (
        ("dev", dev_seeds, dev_weathers),
        ("test", test_seeds, test_weathers),
    ):
        if not weathers:
            print(f"\n[{split}] no strips available, skipping")
            continue
        print(f"\n[{split}] generating...")
        build_valid_set(seeds[0], len(seeds), weathers, HOLDOUT_DIR / split)

        rows = describe(HOLDOUT_DIR / split)
        print(f"\n[{split}] {len(rows)} scenarios")
        seen = {}
        for r in rows:
            seen[r["building"]] = (r["zones"], r["ahus"], r["floors"])
        for b, (z, a, f) in sorted(seen.items()):
            flag = ""
            if isinstance(z, int) and not (MIN_ZONES <= z <= MAX_ZONES):
                flag = f"  <-- outside the stated {MIN_ZONES}-{MAX_ZONES} zone range"
            print(f"  {b:16s} zones={z} ahus={a} floors={f}{flag}")

    print(f"\nholdout scenarios under {HOLDOUT_DIR}/ (git-ignored)")
    print("evaluate with:")
    print(f"  python -m hvaclab.sweep --policies occ "
          f"--scenarios-dir {HOLDOUT_DIR}/dev")


if __name__ == "__main__":
    main()
