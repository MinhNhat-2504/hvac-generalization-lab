# 60-minute hold follow-up

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.
This is an exploratory follow-up after s_night0 failed dev. The three training episodes and control are reused, not new independent evidence.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | s_night12 | 3 | 0.03207875 | 0.03192036 | -0.49% |
| dev screen | s_night12 | 4 | 0.02052920 | 0.02119149 | +3.23% |

Decision:

```json
{
  "unchanged_hold_full_episode_parity": true,
  "training_screen": {
    "s_night12": {
      "n": 3,
      "score": 0.03192036194184934,
      "e_sync": 0.032078747662386616,
      "change_pct": -0.49374034860776295,
      "worst_ratio": 1.0,
      "passes": true
    }
  },
  "selected": "s_night12",
  "dev_screen": {
    "n": 4,
    "score": 0.02119148887833558,
    "e_sync": 0.020529197511646616,
    "change_pct": 3.2260947672856366,
    "worst_ratio": 1.1888508554458765,
    "passes": false
  },
  "stop_reason": "dev gate failed; no full sweep or submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.