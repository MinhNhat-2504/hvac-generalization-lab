# Full-development geometry-scaled zone PI

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched e_sync references.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|
| full dev | geo_p_sqrt | 20 | 0.01668440 | 0.01621940 | -2.79% |
| full dev | geo_pi_sqrt | 20 | 0.01668440 | 0.01609201 | -3.55% |
| full dev | geo_p_linear | 20 | 0.01668440 | 0.01614269 | -3.25% |
| full training | geo_pi_sqrt | 30 | 0.01445394 | 0.01415713 | -2.05% |

Decision:

```json
{
  "unchanged_geometry_full_episode_parity": true,
  "full_dev_calibration": {
    "geo_p_sqrt": {
      "n": 20,
      "score": 0.016219400381487722,
      "e_sync": 0.016684401534295998,
      "change_pct": -2.7870412483925966,
      "worst_ratio": 1.017836517021533,
      "improved": 17,
      "regressed": 3,
      "largest_increases": [
        {
          "scenario": "building_9000/singapore_2",
          "score": 0.017662642074194653,
          "e_sync": 0.017353122803925678,
          "difference": 0.000309519270268975
        },
        {
          "scenario": "building_9000/atlanta_4",
          "score": 0.01571015558265382,
          "e_sync": 0.015520697141682535,
          "difference": 0.00018945844097128552
        },
        {
          "scenario": "building_9003/singapore_2",
          "score": 0.011193280839529876,
          "e_sync": 0.011040115606747886,
          "difference": 0.0001531652327819901
        }
      ],
      "comfort": 0.003003290431451389,
      "energy": 0.00657339670343232,
      "carbon": 0.006642713247962065,
      "passes": true
    },
    "geo_pi_sqrt": {
      "n": 20,
      "score": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_pct": -3.550562189773643,
      "worst_ratio": 1.0479798594645025,
      "improved": 17,
      "regressed": 3,
      "largest_increases": [
        {
          "scenario": "building_9000/singapore_2",
          "score": 0.018185723197328283,
          "e_sync": 0.017353122803925678,
          "difference": 0.0008326003934026049
        },
        {
          "scenario": "building_9001/atlanta_4",
          "score": 0.010282731424952531,
          "e_sync": 0.010201300717759586,
          "difference": 8.143070719294522e-05
        },
        {
          "scenario": "building_9000/atlanta_4",
          "score": 0.01555652196453526,
          "e_sync": 0.015520697141682535,
          "difference": 3.582482285272373e-05
        }
      ],
      "comfort": 0.002745680025322144,
      "energy": 0.006637109362985195,
      "carbon": 0.006709222095138962,
      "passes": true
    },
    "geo_p_linear": {
      "n": 20,
      "score": 0.016142685659097114,
      "e_sync": 0.016684401534295998,
      "change_pct": -3.2468403142021485,
      "worst_ratio": 1.005394300504339,
      "improved": 18,
      "regressed": 2,
      "largest_increases": [
        {
          "scenario": "building_9000/denver_1",
          "score": 0.04170197162090719,
          "e_sync": 0.04147822560759306,
          "difference": 0.0002237460133141314
        },
        {
          "scenario": "building_9001/atlanta_4",
          "score": 0.010246098821484764,
          "e_sync": 0.010201300717759586,
          "difference": 4.479810372517816e-05
        },
        {
          "scenario": "building_9000/fairbanks_3",
          "score": 0.03537513146089218,
          "e_sync": 0.035420502205832455,
          "difference": -4.53707449402721e-05
        }
      ],
      "comfort": 0.0028676897663021303,
      "energy": 0.006577720699270021,
      "carbon": 0.006697275191562093,
      "passes": true
    }
  },
  "ranked_candidates": {
    "1": [
      "geo_pi_sqrt",
      "geo_p_linear",
      "geo_p_sqrt"
    ]
  },
  "full_trials": {
    "geo_pi_sqrt": {
      "training": {
        "n": 30,
        "score": 0.014157125012139818,
        "e_sync": 0.014453943529822781,
        "change_pct": -2.053546958105501,
        "worst_ratio": 1.0311818295267676,
        "improved": 27,
        "regressed": 3,
        "largest_increases": [
          {
            "scenario": "building_train_8/phoenix_4",
            "score": 0.02403539392222161,
            "e_sync": 0.02336711091899175,
            "difference": 0.0006682830032298595
          },
          {
            "scenario": "building_train_2/fairbanks_2",
            "score": 0.017175388967748235,
            "e_sync": 0.016656023676862503,
            "difference": 0.000519365290885733
          },
          {
            "scenario": "building_train_7/singapore_6",
            "score": 0.01396456629508512,
            "e_sync": 0.013724401905017624,
            "difference": 0.0002401643900674967
          }
        ],
        "comfort": 0.001750798378806279,
        "energy": 0.007003089296362975,
        "carbon": 0.005403237337051633,
        "passes": true
      },
      "dev": {
        "n": 20,
        "score": 0.01609201148182927,
        "e_sync": 0.016684401534295998,
        "change_pct": -3.550562189773643,
        "worst_ratio": 1.0479798594645025,
        "improved": 17,
        "regressed": 3,
        "largest_increases": [
          {
            "scenario": "building_9000/singapore_2",
            "score": 0.018185723197328283,
            "e_sync": 0.017353122803925678,
            "difference": 0.0008326003934026049
          },
          {
            "scenario": "building_9001/atlanta_4",
            "score": 0.010282731424952531,
            "e_sync": 0.010201300717759586,
            "difference": 8.143070719294522e-05
          },
          {
            "scenario": "building_9000/atlanta_4",
            "score": 0.01555652196453526,
            "e_sync": 0.015520697141682535,
            "difference": 3.582482285272373e-05
          }
        ],
        "comfort": 0.002745680025322144,
        "energy": 0.006637109362985195,
        "carbon": 0.006709222095138962,
        "passes": true
      }
    }
  },
  "calibration_batch": 1,
  "evaluating_candidate": "geo_pi_sqrt",
  "full_training": {
    "n": 30,
    "score": 0.014157125012139818,
    "e_sync": 0.014453943529822781,
    "change_pct": -2.053546958105501,
    "worst_ratio": 1.0311818295267676,
    "improved": 27,
    "regressed": 3,
    "largest_increases": [
      {
        "scenario": "building_train_8/phoenix_4",
        "score": 0.02403539392222161,
        "e_sync": 0.02336711091899175,
        "difference": 0.0006682830032298595
      },
      {
        "scenario": "building_train_2/fairbanks_2",
        "score": 0.017175388967748235,
        "e_sync": 0.016656023676862503,
        "difference": 0.000519365290885733
      },
      {
        "scenario": "building_train_7/singapore_6",
        "score": 0.01396456629508512,
        "e_sync": 0.013724401905017624,
        "difference": 0.0002401643900674967
      }
    ],
    "comfort": 0.001750798378806279,
    "energy": 0.007003089296362975,
    "carbon": 0.005403237337051633,
    "passes": true
  },
  "full_dev": {
    "n": 20,
    "score": 0.01609201148182927,
    "e_sync": 0.016684401534295998,
    "change_pct": -3.550562189773643,
    "worst_ratio": 1.0479798594645025,
    "improved": 17,
    "regressed": 3,
    "largest_increases": [
      {
        "scenario": "building_9000/singapore_2",
        "score": 0.018185723197328283,
        "e_sync": 0.017353122803925678,
        "difference": 0.0008326003934026049
      },
      {
        "scenario": "building_9001/atlanta_4",
        "score": 0.010282731424952531,
        "e_sync": 0.010201300717759586,
        "difference": 8.143070719294522e-05
      },
      {
        "scenario": "building_9000/atlanta_4",
        "score": 0.01555652196453526,
        "e_sync": 0.015520697141682535,
        "difference": 3.582482285272373e-05
      }
    ],
    "comfort": 0.002745680025322144,
    "energy": 0.006637109362985195,
    "carbon": 0.006709222095138962,
    "passes": true
  },
  "selected": "geo_pi_sqrt",
  "eligible_for_packaging": true,
  "submission_notebook": "experiments/submission/geo_pi_sqrt/submission.ipynb"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Selection optimizes full-cohort mean score. Full-training failures remain recorded while the next dev-ranked candidate is tried.