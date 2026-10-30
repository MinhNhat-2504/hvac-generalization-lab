# Flow-balanced cooling-pressure response

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | cfb_comp | 4 | 0.02564186 | 0.02525925 | -1.49% |
| train screen | cfb_guard | 4 | 0.02564186 | 0.02525925 | -1.49% |
| dev screen | cfb_comp | 4 | 0.01684390 | 0.01720231 | +2.13% |
| dev screen | cfb_guard | 4 | 0.01684390 | 0.01713035 | +1.70% |

Decision:

```json
{
  "unchanged_flow_balance_full_episode_parity": true,
  "training_screen": {
    "cfb_comp": {
      "n": 4,
      "score": 0.02525925159859185,
      "e_sync": 0.025641862347425846,
      "change_pct": -1.4921332298330814,
      "worst_ratio": 1.0,
      "passes": true
    },
    "cfb_guard": {
      "n": 4,
      "score": 0.02525925159859185,
      "e_sync": 0.025641862347425846,
      "change_pct": -1.4921332298330814,
      "worst_ratio": 1.0,
      "passes": true
    }
  },
  "dev_screens": {
    "cfb_comp": {
      "n": 4,
      "score": 0.0172023146522473,
      "e_sync": 0.016843898001117227,
      "change_pct": 2.1278723672293776,
      "worst_ratio": 1.0314725276058219,
      "passes": false
    },
    "cfb_guard": {
      "n": 4,
      "score": 0.017130345672470557,
      "e_sync": 0.016843898001117227,
      "change_pct": 1.700602030090237,
      "worst_ratio": 1.0271169921656071,
      "passes": false
    }
  },
  "screening": "cfb_guard",
  "stop_reason": "all candidate dev gates failed; no submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.