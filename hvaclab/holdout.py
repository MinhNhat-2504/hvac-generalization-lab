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
import random
import re
import shutil
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

# Competition-stated range for the evaluation buildings: 28-85 zones, 3-9 air
# handlers, 2-4 floors. The shipped six land exactly inside it (30-85 zones,
# 3-9 AHUs, 318-430 x 188-301 ft).
MIN_ZONES, MAX_ZONES = 28, 85

# The floor generator's own default footprint is width (400, 700) x height
# (300, 500), which is much larger than any shipped building and produces
# buildings far outside the stated evaluation range -- seed 9000 at the default
# came out with 162 zones and 17 air handlers. Since the scenario generator
# sets `num_ahus = ceil(zones / 10)`, an oversized footprint also multiplies
# simulation cost, and a holdout set that is both unrepresentative and slow is
# the worst of both.
#
# So the footprint is drawn from the shipped six's own range instead. Taken
# from their YAMLs, with a little margin.
FOOTPRINT_WIDTH = (315, 435)
FOOTPRINT_HEIGHT = (185, 305)
FLOOR_CHOICES = (2, 3, 4)


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
    """Run a generator subprocess. Returns whether it succeeded.

    Every path handed to a generator goes in as ``Path.as_posix()``, never
    ``str(Path)``. The scenario generator writes the output directory into the
    gin config as a bare string, and gin parses that string with Python escape
    semantics -- so a Windows path arrives as literal control characters:
    ``dev\\building_9000`` becomes ``dev\\x08uilding_9000`` and
    ``\\floor_plan.npy`` becomes ``\\x0cloor_plan.npy``. The whole set then
    fails at load with ``OSError: [Errno 22] Invalid argument``. Forward
    slashes survive the round trip, and Windows accepts them everywhere here.
    """
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
        "--output-dir", BUILDINGS_DIR.as_posix(),
    ])
    return [f"building_{s:04d}" for s in seeds]


#: gin keys whose value is a filesystem path written by the scenario generator.
_PATH_KEYS = ("floor_plan_filepath", "zone_map_filepath", "metrics_path")


def normalise_gin_paths(scenario_dir: Path) -> None:
    """Rewrite Windows separators in the generated ``config.gin`` paths.

    The scenario generator joins its output path with ``Path`` and writes the
    result into gin as a bare double-quoted string. gin then parses that string
    with Python escape semantics, so on Windows every separator that happens to
    precede a recognised escape letter becomes a control character::

        "...\\dev\\building_9000\\floor_plan.npy"
          ->  "...\\dev\\x08uilding_9000\\x0cloor_plan.npy"

    which fails at load with ``OSError: [Errno 22] Invalid argument``. The
    shipped scenarios do not hit this because they were generated on a
    platform whose separator is already ``/``.

    Passing ``--output-dir`` in posix form is not enough on its own: the
    generator re-joins the path internally before writing it. So the file is
    fixed after the fact, which also keeps the starter kit unmodified.
    """
    gin_file = scenario_dir / "config.gin"
    if not gin_file.exists():
        return
    lines = gin_file.read_text().splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in _PATH_KEYS and "\\" in line:
            lines[i] = line.replace("\\", "/")
            changed = True
    if changed:
        gin_file.write_text("".join(lines))


def count_zones_and_ahus(scenario_dir: Path) -> tuple[int, int]:
    """Realised zone and air-handler counts, read from a built scenario.

    Taken from ``config.gin`` because that is what the simulator consumes.
    ``config.yaml`` carries the *requested* ``num_ahus``, which the scenario
    generator then raises to ``ceil(zones / 10)``, so it is not the answer.
    """
    text = (scenario_dir / "config.gin").read_text()
    zones = len(set(re.findall(r"room_\d+", text)))
    ahus = len(re.findall(r"(?m)^ahu_\d+/AirHandler:", text))
    return zones, ahus


def resize_building(seed: int, rng: random.Random) -> tuple[int, int]:
    """Rewrite a generated building's footprint into the shipped range.

    Returns the (width, height) written. The rest of the generated config --
    room sizes, shape, composite style and the stratified materials block --
    is left exactly as the official generator produced it, so the only thing
    changed is the overall size.
    """
    path = BUILDINGS_DIR / f"building_{seed:04d}.yaml"
    spec = yaml.safe_load(path.read_text())
    width = rng.randint(*FOOTPRINT_WIDTH)
    height = rng.randint(*FOOTPRINT_HEIGHT)
    spec["floor_plan"]["width"] = width
    spec["floor_plan"]["height"] = height
    spec["floor_plan"]["num_floors"] = rng.choice(FLOOR_CHOICES)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return width, height


def rescale_building(seed: int, factor: float) -> None:
    """Scale a building's footprint by ``factor`` in each dimension.

    Zone count goes roughly with floor area, so to move from ``zones`` to
    ``target`` the caller passes ``sqrt(target / zones)``.
    """
    path = BUILDINGS_DIR / f"building_{seed:04d}.yaml"
    spec = yaml.safe_load(path.read_text())
    fp = spec["floor_plan"]
    fp["width"] = max(FOOTPRINT_WIDTH[0] - 60, int(round(fp["width"] * factor)))
    fp["height"] = max(FOOTPRINT_HEIGHT[0] - 60, int(round(fp["height"] * factor)))
    path.write_text(yaml.safe_dump(spec, sort_keys=False))


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
    rng = random.Random(seed_start)

    def floors(s: int) -> bool:
        return _run([
            sys.executable, "-m", "smart_buildings.run_floor_generator",
            "--seeds", str(s), "--no-pdf",
            "--output-dir", BUILDINGS_DIR.as_posix(),
        ])

    def scenarios(name: str) -> bool:
        return _run([
            sys.executable, "-m", "smart_buildings.run_scenario_generator",
            "--buildings", name, "--weathers", *weathers,
            "--output-dir", output_dir.as_posix(),
        ])

    while len(accepted) < count and attempts < max_attempts:
        attempts += 1
        name = f"building_{seed:04d}"

        if not floors(seed):
            print(f"  seed {seed}: floor generation failed, skipping")
            seed += 1
            continue

        resize_building(seed, rng)

        # Build, measure, and if the realised zone count missed the stated
        # range, scale the footprint towards it and build again. Zone count is
        # not a closed form of the footprint -- the generator partitions the
        # plan -- so this is a measured correction rather than a calculation.
        # Each attempt starts from an empty directory for this building, and a
        # building that ends up unusable leaves none behind. Otherwise a
        # rescale that succeeded at the wrong size and then failed on the retry
        # would leave those scenarios on disk, and `describe` -- which reads
        # the directory, not this function's bookkeeping -- would report them
        # as part of the set. That is how a 20-zone building first got into a
        # set whose whole point is to match the stated 28-85 range.
        built = output_dir / name
        ok = False
        for _ in range(3):
            shutil.rmtree(built, ignore_errors=True)
            if not scenarios(name):
                break
            for weather in weathers:
                normalise_gin_paths(built / weather)
            zones, ahus = count_zones_and_ahus(built / weathers[0])
            if MIN_ZONES <= zones <= MAX_ZONES:
                print(f"  seed {seed}: ok -> {name} "
                      f"({zones} zones, {ahus} AHUs)")
                ok = True
                break
            target = (MIN_ZONES + MAX_ZONES) / 2
            print(f"  seed {seed}: {zones} zones outside "
                  f"{MIN_ZONES}-{MAX_ZONES}, rescaling")
            rescale_building(seed, (target / max(zones, 1)) ** 0.5)

        if ok:
            accepted.append(name)
        else:
            shutil.rmtree(built, ignore_errors=True)
            print(f"  seed {seed}: unusable, skipping")

        seed += 1

    if len(accepted) < count:
        print(f"  WARNING: only {len(accepted)}/{count} seeds usable "
              f"after {attempts} attempts")
    return accepted


def describe(output_dir: Path) -> list[dict]:
    """Zone and air-handler counts for everything built under ``output_dir``."""
    rows = []
    for building_dir in sorted(Path(output_dir).iterdir()):
        if not building_dir.is_dir():
            continue
        for scenario_dir in sorted(building_dir.iterdir()):
            if not (scenario_dir / "config.gin").exists():
                continue
            zones, ahus = count_zones_and_ahus(scenario_dir)
            spec = yaml.safe_load((scenario_dir / "config.yaml").read_text()) or {}
            rows.append({
                "building": building_dir.name,
                "weather": scenario_dir.name,
                "zones": zones,
                "ahus": ahus,
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
        seen: dict[str, tuple[int, int, int | None]] = {}
        for r in rows:
            seen[r["building"]] = (r["zones"], r["ahus"], r["floors"])
        for b, (z, a, f) in sorted(seen.items()):
            flag = ""
            if not (MIN_ZONES <= z <= MAX_ZONES):
                flag = f"  <-- outside the stated {MIN_ZONES}-{MAX_ZONES} zones"
            print(f"  {b:16s} zones={z:3d} ahus={a:2d} floors={f}{flag}")

    print(f"\nholdout scenarios under {HOLDOUT_DIR}/ (git-ignored)")
    print("evaluate with:")
    print(f"  python -m hvaclab.sweep --policies occ "
          f"--scenarios-dir {HOLDOUT_DIR}/dev")


if __name__ == "__main__":
    main()
