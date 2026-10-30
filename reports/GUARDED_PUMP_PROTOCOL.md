# Pump-pressure cap after coordinated reheat

This experiment retains e_sync's temperature feedback, zone PI, 288-step
cooling hold and heating-airflow/water interlock. It changes only the final
hot-water differential-pressure request, after all inherited coordination.

| Arm | Normalized pressure cap | Safety guard |
|---|---:|---|
| u_pump_ctrl | None | Exact reference-reproduction control |
| u_pump_cap | 0 | No guard; ablation on the current controller |
| u_pump_guard | 0 | Keep reference pressure near cold edge or high valve demand |

Zero normalized pressure corresponds to 10 bar in the shipped 0–20 bar
configuration, not to zero pressure. The guarded arm leaves the reference
request untouched if any open reheat valve serves a zone at or below the
lower comfort edge plus 0.5 K, its opening is at least 0.8, or temperature is
non-finite. It does not cap when the pump is off or all valves are closed.
Pump power at zero actual flow was already zero in the model.

Lower head also reduces water flow, so lower pump electricity alone does not
establish a lower total score. The guard reads delayed observations and is
not a proof that the actual hidden zone state is safe. The older unconditional
pump-cap failure remains recorded. This ablation checks it again on the newer
coordinated controller and separates the guard from the pressure reduction.

Parameters are fixed before outcomes. Every measured episode has 8,640 steps:

1. Reproduce e_sync on training building_train_7/fairbanks_7 within 1e-10
   mean reward, using the uncapped control.
2. Measure both candidates on training building_train_1/fairbanks_1,
   building_train_7/fairbanks_7 and building_train_8/phoenix_4.
3. Every training-passing candidate gets the four-case dev screen:
   building_9001/fairbanks_3, building_9000/denver_1,
   building_9002/phoenix_2 and building_9003/singapore_2.
4. Gates require a lower matched mean and no case more than 10% worse.
   Among dev-passing candidates, select the lower dev mean for full
   training-30/dev-20 coverage, reusing the selected candidate's episodes.
5. Package a separate notebook only if both complete cohorts also pass the
   gates; verify standalone actions on 576 real-observation steps in each
   of two scenarios and write a submission description. Nothing is sent
   to Kaggle automatically. V4's artifact is retained.

Dev has been used repeatedly for model development; this is exploratory
selection, not untouched confirmation. The test split remains unused.
The cached complete e_sync reference files and all active simulator/policy/
runner/packaging sources and scenario configs are hashed in the protocol.
Original experiment files and prior manuscript numerical results are retained.

The scheduler starts its first simulation only with at least 1.25 GiB
available RAM, and a second only with at least 1.5 GiB available. If RAM is
insufficient, it waits without starting another simulation and resumes
automatically when memory becomes available. At most two simulations run.
Each completed job is saved separately. Check existing processes before
resuming manually:

```powershell
..\.venv\Scripts\python.exe -u -m hvaclab.pump_experiment
```

Progress: `experiments/runs/guarded_pump/status.json`, including
`waiting_for_memory`. Results: `reports/GUARDED_PUMP_REVIEW.md`.
Paper table: `reports/paper/pump_results.tex`, appended to the existing dated
addendum. The PDF is rebuilt at completed stages. A desktop notification
watcher is active for this experiment.
