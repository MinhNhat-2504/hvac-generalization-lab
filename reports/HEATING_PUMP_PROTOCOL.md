# Heating-command pump economy — 28 October 2026

The previous guarded-pump screen finished with 11 valid episodes. The blind
cap regressed 25.19% on training; the guarded cap improved training by 0.42%
but regressed 0.37% on the four-case dev screen. The Singapore dev regression
was predominantly comfort cost while the system was cooling. Those results
remain archived; this is a new exploratory follow-up, not a resumed failed run.

The new controller retains the same normalized-zero pump cap (10 bar under
the shipped 0–20 bar action normalization), 0.5 K lower-band reserve, and
0.8 reheat-valve-opening threshold. It reads only allowed observations and
its own requested actions/state. Its new eligibility gates are:

| Variant | Pressure economy eligibility |
|---|---|
| v_pump_ctrl | Disabled; unchanged V4 control |
| v_pump_warm | Every running AHU must have a strictly positive temperature command |
| v_pump_heat | Same gate, plus a same-step e_sync reheat-airflow command override |

At least one handler must be running. Any running cooling or neutral handler
preserves reference pressure for the shared loop. Command sign is not an
actual supply-temperature measurement. The stricter arm counts e_sync's
command overrides, not physical OFF-to-ON edges. The temperature and valve
guards still apply afterwards. PI, target bands, the full 288-step cooling
hold, water-temperature reset and airflow/water coordination remain unchanged.

No performance gain is guaranteed: pressure also affects water flow, and
the safety reserve uses delayed temperature observations. All constants and
the selection protocol below are fixed before the new outcomes:

1. One uncapped-control episode on training building_train_7/fairbanks_7 must
   reproduce e_sync mean reward within 1e-10.
2. Both candidates run complete 8,640-step episodes on training
   building_train_1/fairbanks_1, building_train_7/fairbanks_7,
   and building_train_8/phoenix_4.
3. Every training-passing candidate is screened on dev
   building_9001/fairbanks_3, building_9000/denver_1,
   building_9002/phoenix_2 and building_9003/singapore_2.
4. Each gate requires a lower matched mean and no scenario more than 10%
   worse. Among dev-passing arms, select the lower dev mean for all 30
   training and 20 dev cases, reusing its completed episodes.
5. Package a separate notebook only after both complete cohorts pass the
   same gates, and verify standalone/reference actions on 576 steps each
   from two actual-observation training rollouts. Save the notebook path
   and submission description. No automatic Kaggle submission occurs.

Dev is repeatedly reused development data for exploratory model selection,
not an untouched confirmation set. The separate test split stays unused.
The V4 notebook and every prior result remain intact.

The scheduler waits for 1.25 GiB available RAM before its first simulation,
and 1.5 GiB before a second. It spaces launches by at least 20 seconds to
allow initial allocations before checking room for another worker. Each
result is saved atomically, with source/config/reference hashes and per-job
logs. Check for a live scheduler before resuming:

```powershell
..\.venv\Scripts\python.exe -u -m hvaclab.heating_pump_experiment
```

Progress: `experiments/runs/heating_pump/status.json`.
Report: `reports/HEATING_PUMP_REVIEW.md`.
Paper table: `reports/paper/heating_pump_results.tex`, added without removing
the older tables; the PDF is rebuilt at completed stages.
A desktop watcher will report final completion or failure.
