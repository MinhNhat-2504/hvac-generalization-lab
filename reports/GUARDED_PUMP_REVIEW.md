# Guarded pump-pressure reset

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | u_pump_cap | 3 | 0.03207875 | 0.04016067 | +25.19% |
| train screen | u_pump_guard | 3 | 0.03207875 | 0.03194406 | -0.42% |
| dev screen | u_pump_guard | 4 | 0.02052920 | 0.02060415 | +0.37% |

Decision:

```json
{
  "unchanged_pump_full_episode_parity": true,
  "training_screen": {
    "u_pump_cap": {
      "n": 3,
      "score": 0.04016066895151928,
      "e_sync": 0.032078747662386616,
      "change_pct": 25.194004997299135,
      "worst_ratio": 1.37479792015675,
      "passes": false
    },
    "u_pump_guard": {
      "n": 3,
      "score": 0.03194406092478714,
      "e_sync": 0.032078747662386616,
      "change_pct": -0.4198628294875717,
      "worst_ratio": 1.0049016367830257,
      "passes": true
    }
  },
  "dev_screens": {
    "u_pump_guard": {
      "n": 4,
      "score": 0.020604154119513857,
      "e_sync": 0.020529197511646616,
      "change_pct": 0.3651219577614606,
      "worst_ratio": 1.031074934384958,
      "passes": false
    }
  },
  "screening": "u_pump_guard",
  "stop_reason": "all candidate dev gates failed; no submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.