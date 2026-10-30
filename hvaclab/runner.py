"""Parallel scenario runner.

``smart_buildings.evaluate`` runs scenarios one at a time and writes charts and
per-step CSVs for each. That is the right default for inspecting a single
rollout, but it makes a full 30-scenario sweep take hours, and a sweep is the
only measurement that corresponds to the leaderboard.

This module runs scenarios in a process pool and keeps only what the score and
the diagnostics need. One scenario is one task, because the simulator holds its
state in module-level gin config and so cannot be shared between threads.

The per-scenario number this records is the mean per-step reward. The
leaderboard metric is the mean of its absolute value across scenarios, so

    score = mean_over_scenarios(|mean_per_step_reward|)

and lower is better. ``summarise`` is the only place that formula appears.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FULL_EPISODE_STEPS = 8640

# Where the shipped scenarios live, relative to the repo root.
SCENARIOS_DIR = Path("smart_buildings/training_scenarios")


@dataclass
class ScenarioResult:
    """One policy on one scenario."""

    building: str
    weather: str
    policy: str
    steps: int
    total_reward: float
    mean_reward: float
    # Reward decomposition, summed over the episode. These are the normalised
    # terms the reward function reports, before their 0.6/0.2/0.2 weights.
    prod_regret: float = 0.0
    energy_cost: float = 0.0
    carbon: float = 0.0
    # Split of the loss by whether anyone was in the building. Comfort is
    # weighted by occupancy, so it is identically zero on unoccupied steps and
    # every unoccupied-step charge is avoidable in principle.
    occupied_steps: int = 0
    reward_occupied: float = 0.0
    reward_unoccupied: float = 0.0
    mean_occupancy: float = 0.0
    seconds: float = 0.0
    error: str = ""

    @property
    def scenario(self) -> str:
        return f"{self.building}/{self.weather}"

    @property
    def score(self) -> float:
        """This scenario's contribution to the leaderboard metric."""
        return abs(self.mean_reward)


def _run_one(task: tuple[str, str, str, int, str]) -> dict[str, Any]:
    """Run one (scenario, policy) pair. Executed in a worker process.

    Imports happen inside the function because the simulator pulls in gin and
    matplotlib, and a worker that is never given work should not pay for them.
    """
    building, weather, policy_key, num_steps, scenarios_dir = task
    t0 = time.time()
    result = ScenarioResult(
        building=building, weather=weather, policy=policy_key, steps=0,
        total_reward=0.0, mean_reward=0.0,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")

        from smart_buildings import evaluate as ev
        from hvaclab.registry import resolve_policy

        config_dir = Path(scenarios_dir) / building / weather
        gin_file = config_dir / "config.gin"
        if not gin_file.exists():
            raise FileNotFoundError(f"no config.gin at {config_dir}")

        env = ev.load_hybrid_environment(gin_file)
        meta = ev.create_building_meta_data(env, config_dir)
        policy = ev.initialize_policy(
            resolve_policy(policy_key), meta, env.action_spec()
        )
        action_spec = env.action_spec()

        # Stepped directly rather than through SimulationTracker. The tracker
        # keeps a per-step row of zone temperatures and a per-step AHU log, and
        # recomputes `get_zone_average_temps()` over the whole grid every step
        # to build it. None of that is needed for a score, and both the memory
        # and the time matter here: a sweep runs 30 of these at once, and the
        # simulator's sparse factorization already fails with
        # SUPERLU_MALLOC when too many live at the same time.
        #
        # Only running scalars are accumulated, so memory is flat in the
        # episode length.
        obs, _ = env.reset()

        total = 0.0
        occupied_steps = 0
        reward_occupied = 0.0
        reward_unoccupied = 0.0
        occupancy_total = 0.0
        prod_regret = 0.0
        energy_cost = 0.0
        carbon = 0.0
        steps = 0

        for _ in range(num_steps):
            action = ev.validate_action(policy.get_action(obs), action_spec)
            obs, reward, terminated, truncated, _ = env.step(action)

            reward = float(reward)
            total += reward
            steps += 1

            # The same response object the scorer reads, so the decomposition
            # here is the scorer's own and not a reconstruction.
            _, rr = env.get_reward_info_and_response()
            prod_regret += float(rr.normalized_productivity_regret)
            energy_cost += float(rr.normalized_energy_cost)
            carbon += float(rr.normalized_carbon_emission)

            occ = float(rr.total_occupancy)
            occupancy_total += occ
            if occ > 0.0:
                occupied_steps += 1
                reward_occupied += reward
            else:
                reward_unoccupied += reward

            if terminated or truncated:
                break

        result.steps = steps
        result.total_reward = total
        result.mean_reward = total / steps if steps else 0.0
        result.mean_occupancy = occupancy_total / steps if steps else 0.0
        result.occupied_steps = occupied_steps
        result.reward_occupied = reward_occupied
        result.reward_unoccupied = reward_unoccupied
        result.prod_regret = prod_regret
        result.energy_cost = energy_cost
        result.carbon = carbon

    except Exception:  # noqa: BLE001 - a failed scenario must not kill the sweep
        result.error = traceback.format_exc(limit=8)

    result.seconds = time.time() - t0
    return asdict(result)


# Peak resident memory of one worker, measured on the largest shipped
# building. The simulator factorizes a sparse finite-difference matrix over the
# building volume, and scipy's SuperLU raises SUPERLU_MALLOC rather than
# degrading when it cannot allocate -- so oversubscribing memory does not run
# slowly, it fails the run outright. Worker count is therefore capped by RAM
# and not only by cores.
WORKER_PEAK_GIB = 0.8


def default_workers(num_tasks: int) -> int:
    """Worker count that fits in both the cores and the free memory."""
    by_cores = max(1, (os.cpu_count() or 4) - 2)

    by_memory = by_cores
    try:
        import psutil

        # Against available rather than total: a desktop running other things
        # does not get to count memory it is not holding.
        free_gib = psutil.virtual_memory().available / 2**30
        by_memory = max(1, int(free_gib * 0.75 / WORKER_PEAK_GIB))
    except Exception:  # noqa: BLE001 - psutil is a convenience, not a dependency
        by_memory = max(1, by_cores // 2)

    return max(1, min(num_tasks, by_cores, by_memory))


def shipped_pairs(scenarios_dir: Path = SCENARIOS_DIR) -> list[tuple[str, str]]:
    """The (building, weather) pairs present on disk, sorted."""
    pairs = []
    for building_dir in sorted(Path(scenarios_dir).iterdir()):
        if not building_dir.is_dir():
            continue
        for weather_dir in sorted(building_dir.iterdir()):
            if (weather_dir / "config.gin").exists():
                pairs.append((building_dir.name, weather_dir.name))
    return pairs


def run_sweep(
    policies: Sequence[str],
    pairs: Sequence[tuple[str, str]] | None = None,
    num_steps: int = FULL_EPISODE_STEPS,
    workers: int | None = None,
    scenarios_dir: Path = SCENARIOS_DIR,
    progress: bool = True,
) -> list[ScenarioResult]:
    """Run every policy on every scenario, in parallel.

    Tasks are ordered longest-first by zone count where that is knowable, which
    for an unbalanced set keeps the pool from finishing with one straggler.
    Here it is approximated by putting all of one policy's work in the queue at
    once and letting the pool interleave.
    """
    if pairs is None:
        pairs = shipped_pairs(scenarios_dir)
    if workers is None:
        workers = default_workers(len(pairs) * len(policies))

    tasks = [
        (b, w, p, num_steps, str(scenarios_dir))
        for p in policies
        for (b, w) in pairs
    ]

    # One thread per worker, set before any child is spawned.
    #
    # The simulator's hot loop is a sparse factorization, and numpy/scipy's
    # BLAS defaults to one thread per core. With N workers that is N x 16
    # threads on 16 cores: the first sweep attempt ran at 1.4 s/step against
    # 0.05 s/step single-threaded, a 27x loss entirely to oversubscription.
    #
    # Each scenario is already an independent task, so thread-level
    # parallelism inside a worker buys nothing and costs cache. Set in the
    # parent because `spawn` children inherit the environment and import
    # numpy fresh, which is the only point these variables are read.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"

    results: list[ScenarioResult] = []
    t0 = time.time()
    def collect(raw_results):
        for i, raw in enumerate(raw_results, 1):
            res = ScenarioResult(**raw)
            results.append(res)
            if progress:
                tag = "FAIL" if res.error else f"{res.score:.5f}"
                print(
                    f"[{i:3d}/{len(tasks)}] {res.policy:18s} {res.scenario:30s} "
                    f"{tag:>10s}  ({res.seconds:5.1f}s)",
                    flush=True,
                )
                if res.error:
                    print(res.error, flush=True)

    if workers == 1:
        # A real serial path needs neither multiprocessing pipes nor a second
        # interpreter. Useful on low-memory machines and restricted Windows
        # sessions where creating a named pipe is denied.
        collect(map(_run_one, tasks))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            collect(pool.imap_unordered(_run_one, tasks))

    if progress:
        print(f"\nsweep of {len(tasks)} runs in {time.time() - t0:.0f}s "
              f"on {workers} workers", flush=True)
    return results


def summarise(results: Sequence[ScenarioResult]) -> dict[str, dict[str, Any]]:
    """Leaderboard score per policy, plus the diagnostics that explain it."""
    out: dict[str, dict[str, Any]] = {}
    for policy in sorted({r.policy for r in results}):
        rows = [r for r in results if r.policy == policy and not r.error]
        if not rows:
            out[policy] = {"score": float("nan"), "n": 0, "failed": True}
            continue
        steps = sum(r.steps for r in rows)
        occ_steps = sum(r.occupied_steps for r in rows)
        out[policy] = {
            # The leaderboard metric.
            "score": float(np.mean([r.score for r in rows])),
            "n": len(rows),
            "worst": max(r.score for r in rows),
            "worst_scenario": max(rows, key=lambda r: r.score).scenario,
            # Where the loss sits. Charges on unoccupied steps buy no comfort,
            # because the comfort term is occupancy-weighted and so zero there.
            "loss_occupied": float(-sum(r.reward_occupied for r in rows)),
            "loss_unoccupied": float(-sum(r.reward_unoccupied for r in rows)),
            "occupied_step_frac": occ_steps / steps if steps else 0.0,
            # Weighted contribution of each reward term to the total loss.
            "w_prod": float(-0.6 * sum(r.prod_regret for r in rows) / steps)
            if steps else 0.0,
            "w_energy": float(0.2 * sum(r.energy_cost for r in rows) / steps)
            if steps else 0.0,
            "w_carbon": float(0.2 * sum(r.carbon for r in rows) / steps)
            if steps else 0.0,
        }
    return out


def print_summary(results: Sequence[ScenarioResult]) -> None:
    """Human-readable sweep report."""
    summary = summarise(results)
    print("\n" + "=" * 100)
    print(f"{'policy':20s}{'SCORE':>10s}{'n':>4s}{'worst':>9s}"
          f"{'comfort':>10s}{'energy':>9s}{'carbon':>9s}"
          f"{'loss@occ':>10s}{'loss@empty':>12s}")
    print("-" * 100)
    for policy, s in sorted(summary.items(), key=lambda kv: kv[1]["score"]):
        if s.get("failed"):
            print(f"{policy:20s}{'FAILED':>10s}")
            continue
        print(
            f"{policy:20s}{s['score']:10.5f}{s['n']:4d}{s['worst']:9.4f}"
            f"{s['w_prod']:10.5f}{s['w_energy']:9.5f}{s['w_carbon']:9.5f}"
            f"{s['loss_occupied']:10.1f}{s['loss_unoccupied']:12.1f}"
        )
    print("=" * 100)
    print("SCORE is the leaderboard metric: mean over scenarios of |mean per-step")
    print("reward|. Lower is better. comfort/energy/carbon are the weighted")
    print("per-step contributions and sum to SCORE. loss@empty is reward given up")
    print("on steps with nobody in the building, where comfort is worth zero.")


def save(results: Sequence[ScenarioResult], path: Path) -> None:
    """Write a sweep to JSON so runs can be compared later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [asdict(r) for r in results],
        "summary": summarise(results),
    }
    path.write_text(json.dumps(payload, indent=2))


def load(path: Path) -> list[ScenarioResult]:
    """Read a sweep back."""
    payload = json.loads(Path(path).read_text())
    return [ScenarioResult(**r) for r in payload["results"]]
