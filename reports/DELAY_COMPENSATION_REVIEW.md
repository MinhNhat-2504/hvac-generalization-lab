# Bounded delay compensation

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | r_half | 3 | 0.03207875 | 0.03274332 | +2.07% |
| train screen | r_full | 3 | 0.03207875 | 0.03198167 | -0.30% |
| dev screen | r_full | 4 | 0.02052920 | 0.02114155 | +2.98% |

Decision:

```json
{
  "zero_blend_full_episode_parity": true,
  "training_screen": {
    "r_half": {
      "n": 3,
      "score": 0.03274332255073485,
      "e_sync": 0.032078747662386616,
      "change_pct": 2.0716983572506287,
      "worst_ratio": 1.0607682618327094,
      "passes": false
    },
    "r_full": {
      "n": 3,
      "score": 0.031981667392806105,
      "e_sync": 0.032078747662386616,
      "change_pct": -0.3026311083033306,
      "worst_ratio": 1.0231851821785727,
      "passes": true
    }
  },
  "selected": "r_full",
  "dev_screen": {
    "n": 4,
    "score": 0.02114154787385878,
    "e_sync": 0.020529197511646616,
    "change_pct": 2.9828265905901485,
    "worst_ratio": 1.0460163186298286,
    "passes": false
  },
  "stop_reason": "dev gate failed; no full sweep or submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.