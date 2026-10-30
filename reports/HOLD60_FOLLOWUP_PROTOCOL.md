# Exploratory follow-up: the remaining one-hour candidate

The preceding scheduled-hold experiment finished with 11 valid full episodes.
Its selected zero-hold candidate regressed 2.44% on the four-case dev screen,
including a 17.59% worst-case regression, and was rejected under that protocol.
The one-hour candidate `s_night12` was not evaluated on dev. Its existing
three-case training mean was 0.03192036194184934 against e_sync
0.032078747662386616, an improvement of 0.49374034860776295%.

This follow-up evaluates that remaining candidate with unchanged parameters:
288-step cooling hold within the PI window (including one-hour prestart),
at most 12 remaining steps outside it. One step is 300 seconds. Genuine
nighttime plant calls, flat bands and heating-water coordination are retained.

Reuse is explicit: three `s_night12` training episodes and one unchanged-control
episode are imported from `experiments/runs/hold_schedule/jobs/`. Every original
source hash, variant parameter set, result fingerprint, task identity and full
8,640-step result is checked first. The new protocol hashes the original
protocol and each reused result file. Imported copies retain `reused_from`
and `original_fingerprint`; original files are not modified. Reused training
scores are not presented as new independent evidence.

The new runs begin directly with the same four development cases:
building_9001/fairbanks_3, building_9000/denver_1,
building_9002/phoenix_2 and building_9003/singapore_2. This is exploratory
model selection after observing a previous candidate's dev failure. The dev
screen is not an untouched confirmation cohort. No test-split cases are used.

The fixed gate requires a lower matched mean and no scenario more than 10%
worse. Passing proceeds to complete training 30 and dev 20 coverage, reusing
the candidate's completed episodes. Only if both full cohorts pass the same
gates does the runner build and verify a separate standalone notebook, save
its path in the decision record, and write its submission description. There
is no automatic Kaggle submission or replacement of the V4 artifact.

The pipeline saves each rollout, updates its own Markdown report and separate
paper table, and recompiles the PDF at completed stages. All earlier paper
tables, measurements and rejected experiments remain retained.

```powershell
# Check that no existing scheduler is running before resuming.
..\.venv\Scripts\python.exe -u -m hvaclab.hold60_experiment
```

Progress: `experiments/runs/hold60_followup/status.json`.
Report: `reports/HOLD60_FOLLOWUP_REVIEW.md`.
Paper table: `reports/paper/hold60_results.tex`.
A desktop completion-notification watcher is active for this run.
