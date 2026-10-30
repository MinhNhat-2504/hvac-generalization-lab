# Bounded cooling-pressure response — 28 October 2026

V4's complete training results identify building_train_7/phoenix_3 as the
largest remaining training comfort contributor. A complete unchanged-control
diagnostic reproduced score 0.016208918072411937. After aligning the diagnostic
hour bins to the observed policy clock, only 12.00% of comfort loss occurred in
the first two scheduled hours. The planned earlier-prestart sweep was stopped
by its predeclared 25% concentration gate, without evaluating its candidates.
The original timestamp-converted diagnostic is retained separately for audit.

On this one reference episode, 96.30% of comfort loss came from overheated
zones. About 92.47% co-occurred with a lagged hot-zone observation, positive
zone-PI cooling demand, a running cooling AHU and a pressure command at or
below the shipped running floor; about 58.07% co-occurred with PI demand at
least 0.8. These are overlapping diagnostic correlations, not demonstrated
causal effects or global-cohort estimates. Directional attribution uses
actual reward-zone data offline; no such data enters the submitted controller.

The new candidates alter only pressure on an already running cooling AHU:

| Variant | Maximum extra-request ceiling |
|---|---:|
| cpi_ctrl | Disabled; exact e_sync action control |
| cpi_400 | normalized -0.6 (400 Pa in shipped normalization) |
| cpi_600 | normalized -0.4 (600 Pa in shipped normalization) |

The PI must be active, and a zone must exceed the comfort upper bound while
its current zone-PI cooling output is at least 0.8. Let d be the largest such
positive excursion and f=min(1,d/1 K). Request -0.8+f*(ceiling+0.8), only if
higher than the inherited request. In the shipped 0–2000 Pa normalization,
-0.8 is the 200 Pa running floor. The response is smooth in temperature error;
it never lowers a higher reference pressure or starts an idle fan.

Missing/non-finite weather or zone temperatures veto the change. A mixed-air
proxy also vetoes it when below the comfort lower bound: requested recirculation
is mapped to [0,1] and capped at 0.7, then combined with mean observed zone
temperature and observed outdoor temperature. This proxy is approximate,
uses delayed sensors, and is not the AHU supply-temperature sensor. More
pressure costs fan power and may overcool other zones; a net benefit is an
empirical question. V4's bands, PI gains, one-hour prestart, pump/water reset
and full 288-step cooling hold remain unchanged.

All constants and the following cases/gates are fixed before candidate
outcomes. Every scored rollout has 8,640 steps:

1. Reproduce e_sync on building_train_7/phoenix_3 within 1e-10 mean reward
   using cpi_ctrl. This is a separate fresh control, not a renamed diagnostic.
2. Evaluate each candidate on building_train_1/fairbanks_1 (cold control),
   building_train_7/phoenix_3, building_train_4/atlanta_5,
   and building_train_7/denver_5 (the three largest residual training-comfort cases).
3. All training-passing arms receive building_9002/phoenix_2,
   building_9002/atlanta_4, building_9002/singapore_2,
   and building_9001/fairbanks_3 on dev.
4. Gates require a lower matched mean with no individual scenario more than
   10% worse. Select the lowest passing dev mean for complete training-30 and
   dev-20 coverage, reusing completed candidate episodes.
5. Only after both full cohorts pass the same gates is a separate standalone
   notebook built, action-parity checked, and recorded with a submission
   description. V4 is retained; no Kaggle submission occurs automatically.

These targeted screens deliberately emphasize the hypothesized mechanism;
their subset gains cannot be reported as global gains. Dev has already been
used for exploratory development; it is not an untouched validation set.
The separate test split stays unused. All previous result files and paper
tables remain retained. The diagnostic file hash and active source/config/
reference hashes are saved in this experiment's protocol.

The scheduler uses RAM gates and at least 20 seconds between interpreter
launches. Check existing processes before resuming:

```powershell
..\.venv\Scripts\python.exe -u -m hvaclab.cooling_pressure_experiment
```

Progress: `experiments/runs/cooling_pressure/status.json`.
Report: `reports/COOLING_PRESSURE_REVIEW.md`.
Paper table: `reports/paper/cooling_pressure_results.tex`, added to the prior
manuscript without deleting earlier measurements. A desktop completion/failure
watcher is active.
