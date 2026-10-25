"""Assemble a self-contained policy file and the Kaggle notebook around it.

The submission runs inside the evaluation sandbox with no network and without
this repository installed, so the policy cannot import ``smart_buildings``. It
has to arrive as one file that depends on nothing but ``numpy`` and ``re``.

Rather than transcribe the controller by hand, which would risk the submitted
policy quietly differing from the one that was measured, this concatenates the
starter kit's own sources:

    smart_buildings/baselines/pid.py        PIDPolicy, imports re + numpy only
    smart_buildings/baselines/pid_plus.py   the four refinements on top of it
    hvaclab/submission/_final_policy.py     our subclass

Only two lines are dropped, both ``import`` statements that the concatenation
makes redundant. Everything else is byte-identical to what was evaluated, and
``verify`` re-runs the assembled file through the local harness and compares it
against the registry variant it is supposed to reproduce.

    python -m hvaclab.submission.build            # write the files
    python -m hvaclab.submission.build --verify   # and check they match
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PID = REPO / "smart_buildings" / "baselines" / "pid.py"
PID_PLUS = REPO / "smart_buildings" / "baselines" / "pid_plus.py"
FINAL = Path(__file__).resolve().parent / "_final_policy.py"

OUT_DIR = REPO / "experiments" / "submission"
OUT_POLICY = OUT_DIR / "submission_policy.py"
OUT_NOTEBOOK = OUT_DIR / "submission.ipynb"

# The two imports the concatenation makes redundant. Matched exactly so that an
# upstream edit which changes them fails loudly here rather than silently
# producing a file that does not import.
_DROP = (
    "from smart_buildings.baselines.pid import PIDPolicy",
    "from hvaclab.submission._shared import PIDWaterResetPolicy",
    # Hoisted into the header instead: a __future__ import is only legal at the
    # top of a file, and concatenation puts this one 900 lines down.
    "from __future__ import annotations",
)

HEADER = '''"""Submission policy for the NeurIPS 2026 Smart Buildings Competition.

GENERATED FILE -- do not edit. Rebuild with:

    python -m hvaclab.submission.build

Assembled from the starter kit's own `pid.py` and `pid_plus.py` plus our
subclass, so that the submitted controller is byte-identical to the one
measured locally. Depends on numpy and re only: the evaluation sandbox has no
network and does not have this repository installed.
"""

from __future__ import annotations

'''


def _strip_imports(text: str) -> str:
    """Remove the import lines listed in ``_DROP``, failing if one is absent."""
    for line in _DROP:
        if line in text:
            text = text.replace(line + "\n", "")
    return text


def assemble() -> str:
    """Return the full text of the self-contained policy module."""
    for path in (PID, PID_PLUS, FINAL):
        if not path.exists():
            raise FileNotFoundError(path)

    parts = [HEADER]
    parts.append("# " + "=" * 74 + "\n")
    parts.append(f"# From {PID.relative_to(REPO).as_posix()} (Apache-2.0, unmodified)\n")
    parts.append("# " + "=" * 74 + "\n\n")
    parts.append(_strip_imports(PID.read_text()))

    parts.append("\n\n# " + "=" * 74 + "\n")
    parts.append(f"# From {PID_PLUS.relative_to(REPO).as_posix()} (Apache-2.0, unmodified)\n")
    parts.append("# " + "=" * 74 + "\n\n")
    parts.append(_strip_imports(PID_PLUS.read_text()))

    parts.append("\n\n# " + "=" * 74 + "\n")
    parts.append("# This entry's controller\n")
    parts.append("# " + "=" * 74 + "\n\n")
    parts.append(_strip_imports(FINAL.read_text()))

    return "".join(parts)


NOTEBOOK_CELL = '''# NeurIPS 2026 Smart Buildings Competition -- submission
#
# {title}
#
# The controller is `pid_water` with three changes, none of them learned:
#
#   1. The comfort band is held around the clock instead of setting back
#      overnight. A building held in band starts the morning with no deviation
#      to recover, and the recovery costs more than the setback saves.
#
#   2. Static pressure is clamped to its floor whenever an air handler is
#      heating. The handler cannot heat -- all heat comes from the VAV reheat
#      coils -- and with recirculation capped at 0.70 every extra cubic metre
#      the fan moves is 0.3 cubic metres of ambient air that must then be
#      heated. In heating, airflow is the load, not just the bill.{cooling_note}
#
#   3. The reheat valve saturates at 0.5 K of shortfall instead of 2.0 K.
#      Throttling it protects a mild climate from overshoot and starves a cold
#      one, where that valve is the only heat source in the building.
#
# Local measurement, full 8640-step episodes, leaderboard metric, lower better:
#
#   {results}

import sys

sys.path.append(
    "/kaggle/input/competitions/neurips-2026-smart-buildings/kaggle_evaluation"
)

from kaggle_evaluation import smart_buildings_inference_server

COMPETITION_DATA_FOLDER = "/kaggle/input/competitions/neurips-2026-smart-buildings/"


{policy_source}


class SubmissionPolicy(smart_buildings_inference_server.Policy):
    """The measured controller, held inside the server's own Policy.

    Composition rather than multiple inheritance, deliberately. Inheriting from
    both would put the server's ``Policy`` last in the MRO, and ``PIDPolicy``
    -- the root of the starter kit chain -- does not call ``super().__init__``,
    so the server's own constructor would never run. Holding the controller as
    an attribute leaves the base class initialised exactly as the official
    template does it, and keeps our ``__init__`` out of that question
    entirely.
    """

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._policy = FinalPolicy(building_meta_data, action_spec)

    def get_action(self, obs):
        return self._policy.get_action(obs)


smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
'''


def build_notebook(policy_source: str, title: str, results: str,
                   cooling_note: str) -> dict:
    """One-cell notebook, matching the shape of the official template."""
    source = NOTEBOOK_CELL.format(
        policy_source=policy_source, title=title, results=results,
        cooling_note=cooling_note,
    )
    return {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def verify() -> bool:
    """Run the assembled policy locally and compare against the registry.

    The simulator is deterministic, so a correct assembly reproduces the
    registry variant's return exactly. Anything other than an exact match means
    the submitted controller is not the one that was measured.
    """
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")

    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy

    spec = importlib.util.spec_from_file_location("submission_policy", OUT_POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scenario = REPO / "smart_buildings" / "training_scenarios" / \
        "building_train_1" / "fairbanks_1"
    steps = 576

    returns = {}
    for label, cls in (("assembled", module.FinalPolicy),
                       ("registry", resolve_policy(module.EQUIVALENT_TO))):
        env = ev.load_hybrid_environment(scenario / "config.gin")
        meta = ev.create_building_meta_data(env, scenario)
        policy = ev.initialize_policy(cls, meta, env.action_spec())
        action_spec = env.action_spec()
        obs, _ = env.reset()
        total = 0.0
        for _ in range(steps):
            action = ev.validate_action(policy.get_action(obs), action_spec)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns[label] = total

    ok = returns["assembled"] == returns["registry"]
    print(f"  assembled : {returns['assembled']!r}")
    print(f"  registry  : {returns['registry']!r}  ({module.EQUIVALENT_TO})")
    print("  MATCH" if ok else "  MISMATCH -- the submission is not what was measured")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="re-run the assembled policy and compare")
    parser.add_argument("--title", default="Airflow is the load, not just the bill")
    parser.add_argument("--results", default="(fill in)")
    parser.add_argument("--cooling-note", default="")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = assemble()
    OUT_POLICY.write_text(source)
    print(f"wrote {OUT_POLICY.relative_to(REPO)} ({len(source.splitlines())} lines)")

    notebook = build_notebook(source, args.title, args.results, args.cooling_note)
    OUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {OUT_NOTEBOOK.relative_to(REPO)}")

    if args.verify:
        print("\nverifying against the local harness...")
        if not verify():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
