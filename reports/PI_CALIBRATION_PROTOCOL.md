# Full-development zone-PI calibration — 28 October 2026

The objective is to beat V4's scenario-mean score, not merely improve a targeted
training subset. Keep the actual e_sync controller architecture: flat bands,
one-hour prestart, original AHU/fan/pump/water behavior, and 288-step cooling
hold. Change only zone PI coefficients and the associated integral-state limit.

The original V4 zone PI has Kp=0.1, Ki=0.005, I_limit=200, giving a maximum
integral contribution Ki*I_limit=1. Older soft/stiff probes changed both gains
on the older plant and did not preserve this quantity. The new prospective
calibration fixes I_limit=1/Ki for every candidate.

| Batch | Variant | Kp | Ki | I_limit |
|---|---|---:|---:|---:|
| Control | pi_cal_ctrl | 0.1 | 0.005 | 200 |
| 1 | pi_cal_slow_i | 0.1 | 0.0025 | 400 |
| 1 | pi_cal_fast_i | 0.1 | 0.01 | 100 |
| 1 | pi_cal_soft_p | 0.05 | 0.005 | 200 |
| 1 | pi_cal_firm_p | 0.2 | 0.005 | 200 |
| 2, if needed | pi_cal_soft_slow | 0.05 | 0.0025 | 400 |
| 2, if needed | pi_cal_soft_fast | 0.05 | 0.01 | 100 |
| 2, if needed | pi_cal_firm_slow | 0.2 | 0.0025 | 400 |
| 2, if needed | pi_cal_firm_fast | 0.2 | 0.01 | 100 |

Fixed before outcomes:

1. Run the control for 8,640 steps on training building_train_7/phoenix_3.
   Require mean-reward agreement with the cached e_sync reference within 1e-10.
2. Evaluate every batch-1 candidate on all 20 dev scenarios. No four-case
   subset or training-screen filter is used to admit candidates.
3. Rank candidates with lower full-dev means and validate all 30 training
   cases in that order. If one fails the training mean, retain its numbers
   and automatically try the next ranked candidate.
4. If none improves both full means, evaluate the fixed batch-2 joint-gain
   corners by the same full-dev and ranked full-training procedure.
5. Package the first candidate improving both complete means by more than
   1e-10 absolute score, verify standalone/reference actions on actual
   observations, and write a validation record and submission description.
   A notebook is not exported merely because an individual case improves.
6. If every tested candidate fails, retain V4 and record the bounded study's
   negative results. This is not a claim that all possible gains are exhausted.

The selection objective is the official scenario-mean loss. There is no
prospective hard 10% worst-scenario cutoff in this study; individual regression
counts and largest increases remain visible. Earlier rejected results are not
reclassified, since their dev means were worse. Dev is reused model-selection
data, and training is a published consistency check; neither is an independent
confirmation set. The separate test split is not used. Local full-cohort
improvement does not guarantee a better Kaggle score.

Runtime is bounded by the fixed candidate set. At most three simulations run,
with at least 1.75 GiB available before launching another and 20 seconds
between launches. Every result is saved atomically; source/config/reference
hashes and dependency versions are recorded. The process lock detects a live
scheduler and recovers a stale lock after a reboot. Interrupted partial jobs
are rerun, while valid complete jobs are reused.

Check state before restarting:

```powershell
..\.venv\Scripts\python.exe -u -m hvaclab.pi_calibration
```

Progress: `experiments/runs/pi_calibration/status.json` (counts per candidate).
Report: `reports/PI_CALIBRATION_REVIEW.md`.
Paper table: `reports/paper/pi_calibration_results.tex`; all previous data
remain intact. A passing notebook's folder also receives `validation.json`
and `SUBMISSION_DESCRIPTION.txt`. No Kaggle submission is sent automatically.
