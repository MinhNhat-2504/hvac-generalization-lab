# V6 geometry sensitivity versus V5 and V4

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched V5 references (V4 also checked).

| Cohort | Variant | n | V5 | Candidate | Change vs V5 |
|---|---|---:|---:|---:|---:|
| full dev | geo6_quarter | 20 | 0.01609201 | 0.01623829 | +0.91% |
| full dev | geo6_threequarter | 20 | 0.01609201 | 0.01604957 | -0.26% |
| full dev | geo6_linear | 20 | 0.01609201 | 0.01616189 | +0.43% |
| full training | geo6_threequarter | 30 | 0.01415713 | 0.01407835 | -0.56% |

Decision:

```json
{
  "unchanged_v5_full_episode_parity": true,
  "full_dev_calibration": {
    "geo6_quarter": {
      "n": 20,
      "score": 0.01623829248378575,
      "v5": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_vs_v4_pct": -2.6738091240086748,
      "change_pct": 0.9090286949003046,
      "worst_ratio": 1.0877146013546155,
      "improved": 7,
      "regressed": 13,
      "largest_increases": [
        {
          "scenario": "building_9000/phoenix_2",
          "score": 0.011209252977753674,
          "v5": 0.010326096728548317,
          "difference": 0.0008831562492053573
        },
        {
          "scenario": "building_9002/singapore_2",
          "score": 0.01885822626877698,
          "v5": 0.018246629630962994,
          "difference": 0.0006115966378139846
        },
        {
          "scenario": "building_9003/phoenix_2",
          "score": 0.007463291604724144,
          "v5": 0.0068614428779659,
          "difference": 0.0006018487267582442
        }
      ],
      "comfort": 0.002956966434473602,
      "energy": 0.006606928249930216,
      "carbon": 0.006674397800770856,
      "passes": false
    },
    "geo6_threequarter": {
      "n": 20,
      "score": 0.016049574002671346,
      "v5": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_vs_v4_pct": -3.8049164084172693,
      "change_pct": -0.2637176788361395,
      "worst_ratio": 1.0364564693279132,
      "improved": 8,
      "regressed": 12,
      "largest_increases": [
        {
          "scenario": "building_9002/atlanta_4",
          "score": 0.017787854961592953,
          "v5": 0.01729562563642562,
          "difference": 0.0004922293251673343
        },
        {
          "scenario": "building_9000/fairbanks_3",
          "score": 0.035766357083966964,
          "v5": 0.035414773989546795,
          "difference": 0.00035158309442016866
        },
        {
          "scenario": "building_9001/fairbanks_3",
          "score": 0.014503455400586063,
          "v5": 0.014152192259339075,
          "difference": 0.0003512631412469873
        }
      ],
      "comfort": 0.002686160054504635,
      "energy": 0.0066276744375296705,
      "carbon": 0.006735739511469646,
      "passes": true
    },
    "geo6_linear": {
      "n": 20,
      "score": 0.016161894197827188,
      "v5": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_vs_v4_pct": -3.1317115893834013,
      "change_pct": 0.43426961307371226,
      "worst_ratio": 1.041427217880612,
      "improved": 7,
      "regressed": 13,
      "largest_increases": [
        {
          "scenario": "building_9002/singapore_2",
          "score": 0.01900253673227173,
          "v5": 0.018246629630962994,
          "difference": 0.0007559071013087358
        },
        {
          "scenario": "building_9000/denver_1",
          "score": 0.042005601249594064,
          "v5": 0.04124969923692813,
          "difference": 0.0007559020126659335
        },
        {
          "scenario": "building_9002/atlanta_4",
          "score": 0.017827728778619788,
          "v5": 0.01729562563642562,
          "difference": 0.0005321031421941687
        }
      ],
      "comfort": 0.002783874060089809,
      "energy": 0.006645505327053379,
      "carbon": 0.006732514814362292,
      "passes": false
    }
  },
  "ranked_candidates": {
    "1": [
      "geo6_threequarter"
    ]
  },
  "full_trials": {
    "geo6_threequarter": {
      "training": {
        "n": 30,
        "score": 0.014078351403852468,
        "v5": 0.014157125012139818,
        "e_sync": 0.014453943529822781,
        "change_vs_v4_pct": -2.59854430173575,
        "change_pct": -0.5564237669710592,
        "worst_ratio": 1.029369641977265,
        "improved": 23,
        "regressed": 7,
        "largest_increases": [
          {
            "scenario": "building_train_8/fairbanks_8",
            "score": 0.0331613193971769,
            "v5": 0.03221517134843705,
            "difference": 0.0009461480487398463
          },
          {
            "scenario": "building_train_4/singapore_3",
            "score": 0.01573016474068454,
            "v5": 0.015477258271979023,
            "difference": 0.00025290646870551603
          },
          {
            "scenario": "building_train_1/singapore_8",
            "score": 0.01430267416767945,
            "v5": 0.014073339168239033,
            "difference": 0.00022933499944041702
          }
        ],
        "comfort": 0.0016661450629854124,
        "energy": 0.00700029911365427,
        "carbon": 0.005411907225751309,
        "passes": true
      },
      "dev": {
        "n": 20,
        "score": 0.016049574002671346,
        "v5": 0.01609201148182927,
        "e_sync": 0.016684401534295998,
        "change_vs_v4_pct": -3.8049164084172693,
        "change_pct": -0.2637176788361395,
        "worst_ratio": 1.0364564693279132,
        "improved": 8,
        "regressed": 12,
        "largest_increases": [
          {
            "scenario": "building_9002/atlanta_4",
            "score": 0.017787854961592953,
            "v5": 0.01729562563642562,
            "difference": 0.0004922293251673343
          },
          {
            "scenario": "building_9000/fairbanks_3",
            "score": 0.035766357083966964,
            "v5": 0.035414773989546795,
            "difference": 0.00035158309442016866
          },
          {
            "scenario": "building_9001/fairbanks_3",
            "score": 0.014503455400586063,
            "v5": 0.014152192259339075,
            "difference": 0.0003512631412469873
          }
        ],
        "comfort": 0.002686160054504635,
        "energy": 0.0066276744375296705,
        "carbon": 0.006735739511469646,
        "passes": true
      }
    }
  },
  "calibration_batch": 1,
  "evaluating_candidate": "geo6_threequarter",
  "full_training": {
    "n": 30,
    "score": 0.014078351403852468,
    "v5": 0.014157125012139818,
    "e_sync": 0.014453943529822781,
    "change_vs_v4_pct": -2.59854430173575,
    "change_pct": -0.5564237669710592,
    "worst_ratio": 1.029369641977265,
    "improved": 23,
    "regressed": 7,
    "largest_increases": [
      {
        "scenario": "building_train_8/fairbanks_8",
        "score": 0.0331613193971769,
        "v5": 0.03221517134843705,
        "difference": 0.0009461480487398463
      },
      {
        "scenario": "building_train_4/singapore_3",
        "score": 0.01573016474068454,
        "v5": 0.015477258271979023,
        "difference": 0.00025290646870551603
      },
      {
        "scenario": "building_train_1/singapore_8",
        "score": 0.01430267416767945,
        "v5": 0.014073339168239033,
        "difference": 0.00022933499944041702
      }
    ],
    "comfort": 0.0016661450629854124,
    "energy": 0.00700029911365427,
    "carbon": 0.005411907225751309,
    "passes": true
  },
  "full_dev": {
    "n": 20,
    "score": 0.016049574002671346,
    "v5": 0.01609201148182927,
    "e_sync": 0.016684401534295998,
    "change_vs_v4_pct": -3.8049164084172693,
    "change_pct": -0.2637176788361395,
    "worst_ratio": 1.0364564693279132,
    "improved": 8,
    "regressed": 12,
    "largest_increases": [
      {
        "scenario": "building_9002/atlanta_4",
        "score": 0.017787854961592953,
        "v5": 0.01729562563642562,
        "difference": 0.0004922293251673343
      },
      {
        "scenario": "building_9000/fairbanks_3",
        "score": 0.035766357083966964,
        "v5": 0.035414773989546795,
        "difference": 0.00035158309442016866
      },
      {
        "scenario": "building_9001/fairbanks_3",
        "score": 0.014503455400586063,
        "v5": 0.014152192259339075,
        "difference": 0.0003512631412469873
      }
    ],
    "comfort": 0.002686160054504635,
    "energy": 0.0066276744375296705,
    "carbon": 0.006735739511469646,
    "passes": true
  },
  "selected": "geo6_threequarter",
  "eligible_for_packaging": true,
  "submission_notebook": "experiments/submission/geo6_threequarter/submission.ipynb"
}
```

Earlier paper sections and V4/V5 artifacts remain unchanged. Change percentages use V5; JSON also reports V4. Selection requires lower full-cohort means than BOTH V5 and V4. Full-training failures remain recorded while the next dev-ranked candidate is tried.