# HVAC Generalization Lab

Transferable HVAC control policies for the
[NeurIPS 2026 Smart Buildings competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/).
Lower competition scores are better.

This repository publishes source code, tests and execution tools. Research
notes, paper files, experiment plans/results and generated submission notebooks
are maintained locally and are excluded from new commits. Earlier Git history
has not been rewritten.

## Source layout

- `smart_buildings/`: supplied simulator and baseline policy sources.
- `hvaclab/policies/`: control policies and experimental extensions.
- `hvaclab/registry.py`: baseline and named policy variants.
- `hvaclab/runner.py`, `hvaclab/sweep.py`: rollout and scoring tools.
- `hvaclab/coverage.py`: checkpointed independent rollout workers.
- `hvaclab/submission/`: standalone policy/notebook assembly and action parity checks.
- `hvaclab/supervisor.py`: background supervision and checkpoint recovery.
- `hvaclab/pipeline_entry.py`, `hvaclab/pipeline_queue.py`: isolated candidate
  registration and sequential experiment scheduling.
- `reports/scripts/`: Python and PowerShell execution helpers (source only).
- `tests/`: controller, packaging, scheduler and supervisor checks.

## Environment and checks

Use the project Python environment with the dependencies declared in
`environment.yml` and `pyproject.toml`. Run from the repository root:

```powershell
python -m unittest discover -s tests -q
python -m hvaclab.sweep --help
python -m hvaclab.submission.experimental --help
```

Simulator rollouts require locally available scenario configurations and data.
Research schedulers additionally require their local frozen plans and matched
reference results; these inputs are intentionally not distributed in the current
source-only tree. Do not treat the absence of local research files as a controller
failure. The source and unit tests can be reviewed independently of those runs.

## Submissions

Standalone notebooks are generated locally under `experiments/submission/`.
Verify action parity before uploading a notebook. A local score improvement
is exploratory evidence and does not establish an organizer score improvement.
The separate local test split is not used for the recent tuning studies.

Keep immutable completed measurement records locally. Preserve earlier paper
data when appending new findings. Never add credentials or private datasets.

## License

Apache-2.0; see `LICENSE` and `NOTICE`. Competition materials belong to their
respective owners. This repository is an independent entry.
