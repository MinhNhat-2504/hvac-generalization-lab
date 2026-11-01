# HVAC Generalization Lab

Updated 2026-10-31 (Asia/Saigon). Actual repository is this directory;
workspace parent contains `.venv`. Run commands here using
`..\.venv\Scripts\python.exe` (PowerShell).

Update 2026-11-01: user screenshot confirms Kaggle Version 7
(`valve8_gentle`) public score **0.01922**, now best, 3.125% below V4.
Older best/pending statements below are historical; current target is <0.01922.

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
- V8 completed 2026-10-31 10:35: 91 full episodes, no errors. Selected
  `valve8_gentle` (valve exponent .75), training mean .0143645138 (-.6187%
  versus V4), dev mean .0164503660 (-1.4027%). Training improves 14 cases,
  regresses 15, ties one; dev improves 12, regresses eight. Comfort penalties
  rise slightly; energy/carbon savings produce the mean improvement.
- Verified notebook: `experiments/submission/valve8_gentle/submission.ipynb`;
  action parity 1,152 real-observation pairs, notebook SHA matches validation,
  variant/source-LF-hash startup marker present. Paper PDF includes both full
  cohort scores. This is an experimental candidate, not a Kaggle improvement.
- User uploaded `valve8_gentle` as Kaggle notebook Version 7; its score is
  pending. Kaggle notebook version numbers differ from internal study labels
  (`geo7_*` was a rejected local study, not this submitted notebook).
- New conditional-curve follow-up: `hvaclab/gated_reheat_calibration.py`,
  outputs `experiments/runs/gated_reheat_calibration`. Test .75 valve boost
  gated by the final current-step positive-SAT running AHU command, a continuous
  cold-zone blend (full at .5 K below the low comfort edge), or both. No labels
  or geometry. Require full means below V4 AND submitted `valve8_gentle`.
- 104 tests passed; old submitted assemblies unchanged. Two fresh full controls
  on building_train_1/fairbanks_1 must reproduce V4 and the global curve before
  60 dev episodes and ranked 30-case training checks. Maximum 152 episodes.
- Current follow-up: first control worker consuming CPU, 0 completed, no errors;
  scheduler may wait for RAM before adding workers. Original 1.75 GiB threshold
  and frozen source hashes preserved. A proposed lower first-worker threshold
  was withdrawn when the original study started; do not assume an r2 run exists.
  Desktop completion watcher is active. Paper additions preserve all old data.
- At most three workers; launch requires 1.75 GiB available RAM and 1 GiB disk.
  Low memory has repeatedly delayed launches. D: previously filled; user
  freed space. Resume matching saved records rather than restarting from zero.
- Desktop notifier: `reports/scripts/notify_delay_completion.ps1`, scoped with
  `-RunRelativePath experiments\runs\split_geometry_calibration`.
  Do not promise unsolicited future chat messages. PC must remain awake.
- V7 builds paper separately and replaces published PDF only after success.
- Next: finish the conditional-curve study, obtain the submitted global-curve
  Kaggle score and check its startup variant/source marker before interpreting
  it. Retain V4 as organizer reference until a lower public score arrives.
  Actual uploaded V6/runtime remains unverified. Preserve all prior results.
- User explicitly requested commits and a push to GitHub. Remote main was
  fetched and verified to be an ancestor of local main; use a normal push,
  never force. Publish source/tests, completed results/provenance and additive
  paper updates. Ignore machine-local logs/locks/notices, test fixtures and
  active V8 result files; publish its frozen protocol only until completion.
- Publication verified: three new commits pushed normally to `origin/main`:
  `2b15ea2` (source/tests), `148501e` (completed results/provenance),
  `8c71a32` (paper/docs). Remote main equals full local HEAD
  `8c71a326ec18f906b378ba87fa6bd0d64b17e0ca`. All 94 tests passed again.
  Active V8 continued at 8/60 dev plus control, three workers, no errors;
  frozen source hashes remained unchanged. Live paper tables and this local
  context can differ from the published snapshot as execution progresses.
- Completed V8 results/paper were subsequently pushed as `aed0551`.
- Conditional-curve code/tests/protocol/paper pushed as `e9fef0e`. Final check:
  one control worker consuming CPU, no errors, no changed frozen source hashes;
  about .97 GiB RAM available, so additional workers wait. Completion notifier
  is watching. Ten gated-policy/workflow tests passed again after setup.

## Independent supervision (2026-11-01)
- User authorized automatic sequential execution of the prepared release study.
  Detached `hold_release_queue` waits for persistence completion and scheduler
  exit, then prepares/supervises `hold_release_calibration`. Child-local entry
  adapter leaves shared registry/builder unchanged. Require full means below
  V7/V4 and any selected persistence winner. Latest training count 11/30;
  three active workers. No second-study rollout started yet.
- 2026-11-02: user explicitly requested restart of persistence study; stopped
  supervisor via stop marker, terminated only its scheduler/worker subtree,
  preserved all 66 results, relaunched independent supervisor. Latest count
  training persist_432 5/30, three workers active, no errors, median ~8.5 min.
- Prepared orthogonal candidate `policies/comfort_hold_release.py`: clear only
  stale cooling hold memory after sustained all-zones-below-centre plus cold
  deficit, retaining V7 valve curve and parent requests/coordination. No registry
  or builder edits, so running study's frozen hashes remain unchanged. No
  performance rollout yet. Prospective arms and next integration/evaluation
  steps in `reports/COMFORT_HOLD_RELEASE_PROTOCOL.md`; do not claim a winner.
- New persistence study: `hvaclab/persistence_calibration.py`, directory
  `experiments/runs/persistence_calibration`, full cooling hold 144/432/576
  steps against V7's 288, global valve exponent .75 unchanged. Fresh full V7
  control; all 20 dev for three arms, ranked 30 training checks. Require both
  complete means below V7 and V4 before packaging. Maximum 151 episodes.
- 116 tests passed; V7 artifact unchanged; paper PDF rebuilt with conditional
  gate negative results and new prospective section. Independent supervisor
  started using `-Study persistence_calibration`; notifier watching. Latest
  setup: waiting for RAM (~.83 GiB versus required 1.75), no rollouts yet.
- Supervisor/launcher now accept an allowlisted study selector (default gated
  reheat, or persistence_calibration). Existing seven supervisor tests pass.
- Conditional-curve study completed 2026-11-01 20:43 (Asia/Saigon): 62 valid
  episodes (two controls plus 60 dev), zero errors. All gates failed against
  submitted valve8_gentle on full dev: heat +.3161%, cold +1.3565%, heat+cold
  +1.1579%. No training launched and no new notebook packaged. Supervisor
  correctly recorded completion and exited; no restart was needed after adoption.
  Kaggle Version 7 / valve8_gentle remains best reported public score .01922.
- User requested robust background supervision. Added `hvaclab/supervisor.py`
  and hidden independent launcher `reports/scripts/start_study_supervisor.ps1`.
  Launched outside the tool session, verified alive and adopting existing
  scheduler/three workers. Current dev count 42/60 plus two controls.
- Every 15s checks exact scheduler/module and resolved worker job-directory;
  waits for orphan workers before restarting, validates frozen source/reference
  hashes and RAM/disk, resumes saved episodes. Logs exit codes and separate
  scheduler output, retry backoff 15–300s, blocks after five consecutive short
  failures. Stop marker ends supervision without terminating the study.
- Seven dedicated tests passed including real child exit code 7, logged crash,
  retry and completion; no real rollout was killed for testing. Frozen study
  sources remain unchanged. Details in `reports/STUDY_SUPERVISOR.md`.
- No automatic reboot/logoff startup or sleep prevention was installed. Sleep
  stops computation; after reboot run the launcher. Existing adopted scheduler
  has no retroactive output log; new supervised launches do.
