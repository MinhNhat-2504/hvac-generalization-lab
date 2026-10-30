# Scheduled cooling hold

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | s_night0 | 3 | 0.03207875 | 0.03156326 | -1.61% |
| train screen | s_night12 | 3 | 0.03207875 | 0.03192036 | -0.49% |
| dev screen | s_night0 | 4 | 0.02052920 | 0.02103063 | +2.44% |

Decision:

```json
{
  "unchanged_hold_full_episode_parity": true,
  "training_screen": {
    "s_night0": {
      "n": 3,
      "score": 0.03156325942652214,
      "e_sync": 0.032078747662386616,
      "change_pct": -1.6069462601524864,
      "worst_ratio": 0.9995552426184009,
      "passes": true
    },
    "s_night12": {
      "n": 3,
      "score": 0.03192036194184934,
      "e_sync": 0.032078747662386616,
      "change_pct": -0.49374034860776295,
      "worst_ratio": 1.0,
      "passes": true
    }
  },
  "selected": "s_night0",
  "dev_screen": {
    "n": 4,
    "score": 0.021030630352159817,
    "e_sync": 0.020529197511646616,
    "change_pct": 2.4425350295778747,
    "worst_ratio": 1.175899588242792,
    "passes": false
  },
  "stop_reason": "dev gate failed; no full sweep or submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.