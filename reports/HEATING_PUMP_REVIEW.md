# Heating-command pump economy

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | v_pump_warm | 3 | 0.03207875 | 0.03203897 | -0.12% |
| train screen | v_pump_heat | 3 | 0.03207875 | 0.03203897 | -0.12% |
| dev screen | v_pump_warm | 4 | 0.02052920 | 0.02062192 | +0.45% |
| dev screen | v_pump_heat | 4 | 0.02052920 | 0.02062192 | +0.45% |

Decision:

```json
{
  "unchanged_heating_pump_full_episode_parity": true,
  "training_screen": {
    "v_pump_warm": {
      "n": 3,
      "score": 0.03203896980150482,
      "e_sync": 0.032078747662386616,
      "change_pct": -0.12400066642388463,
      "worst_ratio": 1.0049016367830257,
      "passes": true
    },
    "v_pump_heat": {
      "n": 3,
      "score": 0.03203896980150482,
      "e_sync": 0.032078747662386616,
      "change_pct": -0.12400066642388463,
      "worst_ratio": 1.0049016367830257,
      "passes": true
    }
  },
  "dev_screens": {
    "v_pump_warm": {
      "n": 4,
      "score": 0.02062191711735891,
      "e_sync": 0.020529197511646616,
      "change_pct": 0.4516474921130831,
      "worst_ratio": 1.0301948459191803,
      "passes": false
    },
    "v_pump_heat": {
      "n": 4,
      "score": 0.02062191711735891,
      "e_sync": 0.020529197511646616,
      "change_pct": 0.4516474921130831,
      "worst_ratio": 1.0301948459191803,
      "passes": false
    }
  },
  "screening": "v_pump_heat",
  "stop_reason": "all candidate dev gates failed; no submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.