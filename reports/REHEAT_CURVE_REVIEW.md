# V4 reheat-valve characteristic study

Status: running; stage: full_dev_calibration.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|

Decision:

```json
{
  "unchanged_v4_full_episode_parity": true,
  "full_dev_calibration": {},
  "ranked_candidates": {},
  "full_trials": {},
  "calibration_batch": 1
}
```

Earlier paper sections and the e_sync public score remain unchanged. Selection optimizes full-cohort mean score. Full-training failures remain recorded while the next dev-ranked candidate is tried.