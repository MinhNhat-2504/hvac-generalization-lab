# Bounded causal delay compensation — protocol fixed before rollouts

The reference is `e_sync` (Kaggle V4 public score 0.01984). Full local reference
cohorts are training 30, score 0.014453943529822781, and dev 20, score
0.01668440. Candidate scores must be measured on matching complete episodes;
short diagnostic traces are not comparable policy scores.

The causal per-zone ridge model is the shadow model, updated online within
each new episode. At observation t, it learns the preceding target and
forecasts observation t+1 using observations available through t and action
t−1. State, weights and training history are not shared across buildings.
No future actions, true thermal state or building/weather identifiers enter
the controller. The aggregate AHU and hot-water loops keep their original
observation input. Only PI zone feedback receives the temperature correction.

- `r_zero`: zero blend, action-level and one full-episode reproduction control.
- `r_half`: half of the forecast, bounded to a correction of ±0.5 K.
- `r_full`: full forecast, bounded to a correction of ±1 K.
- Both nonzero blends start after 288 calls. Prior is identity ridge covariance;
  there is no forgetting-factor or gain search in this screen.

All episodes run 8,640 steps. The seven initial runs are one `r_zero` episode
on training `building_train_7/fairbanks_7`, plus both nonzero candidates on:

| Building | Weather | Role |
|---|---|---|
| building_train_1 | fairbanks_1 | Deep-cold heating |
| building_train_7 | fairbanks_7 | Shoulder-season heating |
| building_train_8 | phoenix_4 | Hot-weather cooling |

Advance only if zero blend reproduces the cached reference mean reward within
1e-10 and a candidate lowers the three-case mean without making any case more
than 10% worse. If both pass, select the lower training-screen mean. Test only
that selected candidate on this predeclared dev screen:

| Building | Weather |
|---|---|
| building_9001 | fairbanks_3 |
| building_9000 | denver_1 |
| building_9002 | phoenix_2 |
| building_9003 | singapore_2 |

The dev gate again requires a lower matched mean and no case more than 10%
worse. Only a passing candidate proceeds to all 30 training and 20 dev cases,
reusing its completed screen episodes. The test split is untouched. Failed or
short rollouts cannot pass a gate; inspect their logs and resume the experiment.
The two candidate variants will not be changed during this experiment.

The runner saves each result atomically, records attempts separately, freezes
source/config/reference hashes, and uses at most two interpreters with a RAM
check before launching the second. Check existing processes before resuming:

```powershell
..\.venv\Scripts\python.exe -u -m hvaclab.delay_experiment
```

Progress: `experiments/runs/delay_compensation/status.json`.
Results: `reports/DELAY_COMPENSATION_REVIEW.md`. Paper rows are maintained in
`reports/paper/delay_results.tex`, with the PDF rebuilt at completed stages.
All original manuscript text and historical numerical results remain intact;
the pre-update source and PDF are archived in `reports/paper/history/`.
There is no automatic Kaggle submission or replacement of Version 4.
