# Queued Version 8 follow-up

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched submitted-curve references; V4 is checked separately.

| Cohort | Variant | n | Submitted curve | Candidate | Change |
|---|---|---:|---:|---:|---:|
| full dev | curve10_half | 20 | 0.01645037 | 0.01626602 | -1.12% |
| full dev | curve10_mid | 20 | 0.01645037 | 0.01631389 | -0.83% |
| full dev | curve10_light | 20 | 0.01645037 | 0.01628564 | -1.00% |

Decision:

```json
{
  "unchanged_curve_full_episode_parity": true,
  "full_dev_calibration": {
    "curve10_half": {
      "n": 20,
      "score": 0.016266015774698467,
      "submitted_curve": 0.01645036602119071,
      "winner_scores": {
        "persistence_calibration:persist_432": 0.016253459275641348
      },
      "winner_score": 0.016253459275641348,
      "e_sync": 0.016684401534295998,
      "change_vs_v4_pct": -2.507646191189472,
      "change_pct": -1.1206452564932023,
      "worst_ratio": 1.0637263119734794,
      "improved": 13,
      "regressed": 7,
      "largest_increases": [
        {
          "scenario": "building_9000/phoenix_2",
          "score": 0.010877802782405655,
          "submitted_curve": 0.01022612927776939,
          "difference": 0.0006516735046362652
        },
        {
          "scenario": "building_9003/fairbanks_3",
          "score": 0.011070744178009675,
          "submitted_curve": 0.010622402266435486,
          "difference": 0.00044834191157418907
        },
        {
          "scenario": "building_9003/phoenix_2",
          "score": 0.008414650494526549,
          "submitted_curve": 0.008026747518960906,
          "difference": 0.0003879029755656429
        }
      ],
      "comfort": 0.0033465586912164737,
      "energy": 0.006502396511738395,
      "carbon": 0.006417060573187231,
      "passes": false
    },
    "curve10_mid": {
      "n": 20,
      "score": 0.016313891034764327,
      "submitted_curve": 0.01645036602119071,
      "winner_scores": {
        "persistence_calibration:persist_432": 0.016253459275641348
      },
      "winner_score": 0.016253459275641348,
      "e_sync": 0.016684401534295998,
      "change_vs_v4_pct": -2.220699967991413,
      "change_pct": -0.829616716434034,
      "worst_ratio": 1.0656798221845132,
      "improved": 12,
      "regressed": 8,
      "largest_increases": [
        {
          "scenario": "building_9003/fairbanks_3",
          "score": 0.01132007975846734,
          "submitted_curve": 0.010622402266435486,
          "difference": 0.0006976774920318535
        },
        {
          "scenario": "building_9000/phoenix_2",
          "score": 0.010769453821399426,
          "submitted_curve": 0.01022612927776939,
          "difference": 0.0005433245436300364
        },
        {
          "scenario": "building_9001/phoenix_2",
          "score": 0.007107268777714795,
          "submitted_curve": 0.006672974874527321,
          "difference": 0.0004342939031874741
        }
      ],
      "comfort": 0.003324926017353784,
      "energy": 0.006507126799358903,
      "carbon": 0.006481838219517398,
      "passes": false
    },
    "curve10_light": {
      "n": 20,
      "score": 0.0162856350655639,
      "submitted_curve": 0.01645036602119071,
      "winner_scores": {
        "persistence_calibration:persist_432": 0.016253459275641348
      },
      "winner_score": 0.016253459275641348,
      "e_sync": 0.016684401534295998,
      "change_vs_v4_pct": -2.3900555732394912,
      "change_pct": -1.0013817043013407,
      "worst_ratio": 1.0351878371492422,
      "improved": 10,
      "regressed": 10,
      "largest_increases": [
        {
          "scenario": "building_9000/denver_1",
          "score": 0.04076269989381032,
          "submitted_curve": 0.040317059829980434,
          "difference": 0.00044564006382988297
        },
        {
          "scenario": "building_9001/denver_1",
          "score": 0.017719684838512936,
          "submitted_curve": 0.01738076125478545,
          "difference": 0.0003389235837274844
        },
        {
          "scenario": "building_9003/fairbanks_3",
          "score": 0.010922688069780617,
          "submitted_curve": 0.010622402266435486,
          "difference": 0.00030028580334513083
        }
      ],
      "comfort": 0.0032111958013775694,
      "energy": 0.0065119159832180366,
      "carbon": 0.006562523279704751,
      "passes": false
    }
  },
  "ranked_candidates": {
    "1": []
  },
  "full_trials": {},
  "calibration_batch": 1,
  "eligible_for_packaging": false,
  "stop_reason": "no candidate beat all reference means on both cohorts; retain existing submissions"
}
```

Earlier paper sections and artifacts remain unchanged. Percentages compare with the submitted curve; JSON also reports V4. Selection requires full-cohort means below BOTH V4 and the submitted curve. Full-training failures remain recorded while the next dev-ranked candidate is tried.