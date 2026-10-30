# HVAC Generalization Lab

Updated 2026-10-31 (Asia/Saigon). Actual repository is this directory;
workspace parent contains `.venv`. Run commands here using
`..\.venv\Scripts\python.exe` (PowerShell).

## Goal and architecture
- Optimize a transferable HVAC policy for the Kaggle NeurIPS 2026 Smart
  Buildings competition. Lower scores are better. Aim to beat V4 on Kaggle.
- Python/NumPy controllers in `hvaclab/policies`, parameter variants in
  `hvaclab/registry.py`; supplied simulator in `smart_buildings`.
- `hvaclab/runner.py` evaluates full 8,640-step episodes (5 minutes per step).
  Coverage workers use independent subprocesses and JSON job/result files.
- `hvaclab/submission/experimental.py` builds standalone policy notebooks and
  verifies action parity. Do not automatically submit/upload to Kaggle.

## Verified results and decisions
- User screenshots: V4/e_sync public 0.01984, V5/geo_pi_sqrt 0.01996,
  V6/geo6_threequarter 0.03833. V4 remains best among these submissions.
- V5/V6 improved local training/dev means but failed to improve public score.
  Local improvements are exploratory, not evidence of organizer superiority.
- V6 local notebook hashes and 500 wrapper action pairs were audited; the
  actual uploaded Kaggle notebook/runtime has not been verified. V6 matching
  V3's displayed score does not establish a wrong upload or a root cause.
- Preserve old results and V4/V5/V6 artifacts. Paper changes must be additive.
  Original paper archived in `reports/paper/history`; append to
  `reports/paper/updates_2026_10_27.tex` and separate study result tables.

## Current V7 study
- `hvaclab/split_geometry_calibration.py`; run directory
  `experiments/runs/split_geometry_calibration`.
- Separate P/I area exponents: (.75,.5), (.75,1), (.5,.75). Equal exponents
  (.75,.75) reproduce V6. Plant coordination and schedules stay unchanged.
- Completed 2026-10-31 03:48 (Asia/Saigon): 91 full episodes, no failures.
  Only `geo7_strong_i` passed dev (0.0160358888, -0.0853% versus V6), but
  training failed (0.0142295906, +1.0743% versus V6, +0.5119% versus V5).
  No V7 notebook was packaged; all three split-gain rules rejected.
- 87 unit tests passed before study launch. Source/config/reference hashes
  are frozen: do not edit run dependencies while the study is active.
- Packaging requires lower full means than V6, V5 and V4 on all 20 dev and
  all 30 training scenarios, then standalone verification. Test split unused.

## Operational constraints and next steps
- New prospective V8 study: `hvaclab/reheat_curve_calibration.py`, outputs
  `experiments/runs/reheat_curve_calibration`. Return to V4 with no geometry;
  change only active-PI valve opening h to h**p, p=.5/.75/1.5, preserving
  zero/full authority. PI, damper and plant settings remain V4. Require lower
  full dev/training means than V4 before packaging; no public-score guarantee.
- Implemented and prepared 2026-10-31: 94 tests passed; seven relevant tests
  passed again after final baseline-validation/manifest changes. Old V4/V5/V6
  artifacts match their assemblies. Full generated notebook-wrapper tests
  check actions and the new runtime variant/source-LF-hash log marker.
- Latest V8 check (2026-10-31 04:24): running full dev, 6/60 dev rollouts
  plus one matching V4 control complete; three active workers, no failures.
  Read live status/processes for newer counts. Its desktop notifier is watching.
- At most three workers; launch requires 1.75 GiB available RAM and 1 GiB disk.
  Low memory has repeatedly delayed launches. D: previously filled; user
  freed space. Resume matching saved records rather than restarting from zero.
- Desktop notifier: `reports/scripts/notify_delay_completion.ps1`, scoped with
  `-RunRelativePath experiments\runs\split_geometry_calibration`.
  Do not promise unsolicited future chat messages. PC must remain awake.
- V7 builds paper separately and replaces published PDF only after success.
- Next: run the V4-based valve study when memory permits, retain V4 as the
  organizer reference, and verify runtime variant/source markers in future
  Kaggle logs. Actual uploaded V6/runtime remains unverified. V7 negative
  results and the new V8 protocol remain in separate paper additions.
- User explicitly requested commits and a push to GitHub. Remote main was
  fetched and verified to be an ancestor of local main; use a normal push,
  never force. Publish source/tests, completed results/provenance and additive
  paper updates. Ignore machine-local logs/locks/notices, test fixtures and
  active V8 result files; publish its frozen protocol only until completion.
