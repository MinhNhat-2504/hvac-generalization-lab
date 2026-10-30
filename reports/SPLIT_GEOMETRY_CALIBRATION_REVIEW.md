# V7 split geometry sensitivity versus V6, V5 and V4

Status: complete; stage: finished.

All scores use complete 8,640-step episodes against matched V6 references (V5 and V4 also checked).

| Cohort | Variant | n | V6 | Candidate | Change vs V6 |
|---|---|---:|---:|---:|---:|
| full dev | geo7_soft_i | 20 | 0.01604957 | 0.01605772 | +0.05% |
| full dev | geo7_strong_i | 20 | 0.01604957 | 0.01603589 | -0.09% |
| full dev | geo7_soft_p | 20 | 0.01604957 | 0.01613853 | +0.55% |
| full training | geo7_strong_i | 30 | 0.01407835 | 0.01422959 | +1.07% |

Decision:

```json
{
  "unchanged_v6_full_episode_parity": true,
  "full_dev_calibration": {
    "geo7_soft_i": {
      "n": 20,
      "score": 0.016057723070860255,
      "v6": 0.016049574002671346,
      "v5": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_vs_v5_pct": -0.21307722162473564,
      "change_vs_v4_pct": -3.7560739721323566,
      "change_pct": 0.05077435817020248,
      "worst_ratio": 1.045271815532142,
      "improved": 12,
      "regressed": 8,
      "largest_increases": [
        {
          "scenario": "building_9000/singapore_2",
          "score": 0.01785269032403081,
          "v6": 0.017079471634793963,
          "difference": 0.0007732186892368473
        },
        {
          "scenario": "building_9002/phoenix_2",
          "score": 0.013253272405375827,
          "v6": 0.012862294589251633,
          "difference": 0.0003909778161241932
        },
        {
          "scenario": "building_9003/atlanta_4",
          "score": 0.010550869719426082,
          "v6": 0.010207155096646008,
          "difference": 0.0003437146227800742
        }
      ],
      "comfort": 0.0027306377331257933,
      "energy": 0.006628933326990541,
      "carbon": 0.006698152010338701,
      "passes": false
    },
    "geo7_strong_i": {
      "n": 20,
      "score": 0.01603588875214409,
      "v6": 0.016049574002671346,
      "v5": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_vs_v5_pct": -0.3487614320220378,
      "change_vs_v4_pct": -3.886940630257796,
      "change_pct": -0.08526862161560844,
      "worst_ratio": 1.03182877775908,
      "improved": 9,
      "regressed": 11,
      "largest_increases": [
        {
          "scenario": "building_9001/singapore_2",
          "score": 0.011534112161105217,
          "v6": 0.011178319901248478,
          "difference": 0.00035579225985673917
        },
        {
          "scenario": "building_9002/singapore_2",
          "score": 0.018811440129828504,
          "v6": 0.018468339109113666,
          "difference": 0.000343101020714838
        },
        {
          "scenario": "building_9003/fairbanks_3",
          "score": 0.011035339424953843,
          "v6": 0.010864545325683737,
          "difference": 0.00017079409927010618
        }
      ],
      "comfort": 0.00271423194430678,
      "energy": 0.006614517655978052,
      "carbon": 0.006707139151689809,
      "passes": true
    },
    "geo7_soft_p": {
      "n": 20,
      "score": 0.01613853474575543,
      "v6": 0.016049574002671346,
      "v5": 0.01609201148182927,
      "e_sync": 0.016684401534295998,
      "change_vs_v5_pct": 0.28910782209352615,
      "change_vs_v4_pct": -3.2717193206990403,
      "change_pct": 0.55428725441109,
      "worst_ratio": 1.05093492055093,
      "improved": 10,
      "regressed": 10,
      "largest_increases": [
        {
          "scenario": "building_9000/singapore_2",
          "score": 0.017944595379325717,
          "v6": 0.017079471634793963,
          "difference": 0.0008651237445317546
        },
        {
          "scenario": "building_9002/singapore_2",
          "score": 0.01932625921620691,
          "v6": 0.018468339109113666,
          "difference": 0.0008579201070932421
        },
        {
          "scenario": "building_9002/phoenix_2",
          "score": 0.01351743454225782,
          "v6": 0.012862294589251633,
          "difference": 0.0006551399530061869
        }
      ],
      "comfort": 0.0027745336457272074,
      "energy": 0.006634246053190114,
      "carbon": 0.006729755047105249,
      "passes": false
    }
  },
  "ranked_candidates": {
    "1": [
      "geo7_strong_i"
    ]
  },
  "full_trials": {
    "geo7_strong_i": {
      "training": {
        "n": 30,
        "score": 0.014229590608079884,
        "v6": 0.014078351403852468,
        "v5": 0.014157125012139818,
        "e_sync": 0.014453943529822781,
        "change_vs_v5_pct": 0.5118666104729996,
        "change_vs_v4_pct": -1.5521917688414266,
        "change_pct": 1.0742678591332044,
        "worst_ratio": 1.0827231348539337,
        "improved": 12,
        "regressed": 18,
        "largest_increases": [
          {
            "scenario": "building_train_8/phoenix_4",
            "score": 0.025075708203695722,
            "v6": 0.023159852594337147,
            "difference": 0.001915855609358575
          },
          {
            "scenario": "building_train_8/fairbanks_8",
            "score": 0.0342656755274262,
            "v6": 0.0331613193971769,
            "difference": 0.0011043561302493052
          },
          {
            "scenario": "building_train_4/atlanta_5",
            "score": 0.01516303305381278,
            "v6": 0.014462726338998285,
            "difference": 0.0007003067148144953
          }
        ],
        "comfort": 0.001752409618764239,
        "energy": 0.007025951238595383,
        "carbon": 0.005451229750712311,
        "passes": false
      },
      "dev": {
        "n": 20,
        "score": 0.01603588875214409,
        "v6": 0.016049574002671346,
        "v5": 0.01609201148182927,
        "e_sync": 0.016684401534295998,
        "change_vs_v5_pct": -0.3487614320220378,
        "change_vs_v4_pct": -3.886940630257796,
        "change_pct": -0.08526862161560844,
        "worst_ratio": 1.03182877775908,
        "improved": 9,
        "regressed": 11,
        "largest_increases": [
          {
            "scenario": "building_9001/singapore_2",
            "score": 0.011534112161105217,
            "v6": 0.011178319901248478,
            "difference": 0.00035579225985673917
          },
          {
            "scenario": "building_9002/singapore_2",
            "score": 0.018811440129828504,
            "v6": 0.018468339109113666,
            "difference": 0.000343101020714838
          },
          {
            "scenario": "building_9003/fairbanks_3",
            "score": 0.011035339424953843,
            "v6": 0.010864545325683737,
            "difference": 0.00017079409927010618
          }
        ],
        "comfort": 0.00271423194430678,
        "energy": 0.006614517655978052,
        "carbon": 0.006707139151689809,
        "passes": true
      }
    }
  },
  "calibration_batch": 1,
  "evaluating_candidate": "geo7_strong_i",
  "full_training": {
    "n": 30,
    "score": 0.014229590608079884,
    "v6": 0.014078351403852468,
    "v5": 0.014157125012139818,
    "e_sync": 0.014453943529822781,
    "change_vs_v5_pct": 0.5118666104729996,
    "change_vs_v4_pct": -1.5521917688414266,
    "change_pct": 1.0742678591332044,
    "worst_ratio": 1.0827231348539337,
    "improved": 12,
    "regressed": 18,
    "largest_increases": [
      {
        "scenario": "building_train_8/phoenix_4",
        "score": 0.025075708203695722,
        "v6": 0.023159852594337147,
        "difference": 0.001915855609358575
      },
      {
        "scenario": "building_train_8/fairbanks_8",
        "score": 0.0342656755274262,
        "v6": 0.0331613193971769,
        "difference": 0.0011043561302493052
      },
      {
        "scenario": "building_train_4/atlanta_5",
        "score": 0.01516303305381278,
        "v6": 0.014462726338998285,
        "difference": 0.0007003067148144953
      }
    ],
    "comfort": 0.001752409618764239,
    "energy": 0.007025951238595383,
    "carbon": 0.005451229750712311,
    "passes": false
  },
  "full_dev": {
    "n": 20,
    "score": 0.01603588875214409,
    "v6": 0.016049574002671346,
    "v5": 0.01609201148182927,
    "e_sync": 0.016684401534295998,
    "change_vs_v5_pct": -0.3487614320220378,
    "change_vs_v4_pct": -3.886940630257796,
    "change_pct": -0.08526862161560844,
    "worst_ratio": 1.03182877775908,
    "improved": 9,
    "regressed": 11,
    "largest_increases": [
      {
        "scenario": "building_9001/singapore_2",
        "score": 0.011534112161105217,
        "v6": 0.011178319901248478,
        "difference": 0.00035579225985673917
      },
      {
        "scenario": "building_9002/singapore_2",
        "score": 0.018811440129828504,
        "v6": 0.018468339109113666,
        "difference": 0.000343101020714838
      },
      {
        "scenario": "building_9003/fairbanks_3",
        "score": 0.011035339424953843,
        "v6": 0.010864545325683737,
        "difference": 0.00017079409927010618
      }
    ],
    "comfort": 0.00271423194430678,
    "energy": 0.006614517655978052,
    "carbon": 0.006707139151689809,
    "passes": true
  },
  "eligible_for_packaging": false,
  "stop_reason": "no split rule beat V6, V5 and V4 on both full means; retain existing submissions"
}
```

Earlier paper sections and V4/V5/V6 artifacts remain unchanged. Change percentages use V6; JSON also reports V5 and V4. Selection requires lower full-cohort means than V6, V5 and V4. Full-training failures remain recorded while the next dev-ranked candidate is tried.