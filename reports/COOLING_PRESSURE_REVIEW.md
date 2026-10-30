# Pressure response for saturated hot zones

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| train screen | cpi_400 | 4 | 0.02564186 | 0.02541995 | -0.87% |
| train screen | cpi_600 | 4 | 0.02564186 | 0.02544590 | -0.76% |
| dev screen | cpi_400 | 4 | 0.01684390 | 0.01741959 | +3.42% |
| dev screen | cpi_600 | 4 | 0.01684390 | 0.01766523 | +4.88% |

Decision:

```json
{
  "unchanged_cooling_pressure_full_episode_parity": true,
  "training_screen": {
    "cpi_400": {
      "n": 4,
      "score": 0.025419953574464093,
      "e_sync": 0.025641862347425846,
      "change_pct": -0.8654159746865342,
      "worst_ratio": 1.0107855478475052,
      "passes": true
    },
    "cpi_600": {
      "n": 4,
      "score": 0.025445902676643605,
      "e_sync": 0.025641862347425846,
      "change_pct": -0.7642177784403903,
      "worst_ratio": 1.002771555651843,
      "passes": true
    }
  },
  "dev_screens": {
    "cpi_400": {
      "n": 4,
      "score": 0.017419593331687272,
      "e_sync": 0.016843898001117227,
      "change_pct": 3.4178272186869085,
      "worst_ratio": 1.067210479046677,
      "passes": false
    },
    "cpi_600": {
      "n": 4,
      "score": 0.017665228136610492,
      "e_sync": 0.016843898001117227,
      "change_pct": 4.876128645749267,
      "worst_ratio": 1.1102851652941834,
      "passes": false
    }
  },
  "screening": "cpi_600",
  "stop_reason": "all candidate dev gates failed; no submission change"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.