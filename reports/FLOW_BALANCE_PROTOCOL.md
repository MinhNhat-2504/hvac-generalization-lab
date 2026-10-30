# Flow-balanced cooling pressure — 28 October 2026

The uncompensated cooling-pressure experiment finished with 17 valid episodes.
Both arms improved the targeted training mean but regressed dev: 400 Pa gave
-0.87% training / +3.42% dev; 600 Pa gave -0.76% / +4.88%. Phoenix dev comfort
improved while Atlanta/Singapore comfort worsened. Cross-zone airflow coupling
is a candidate explanation, not an established causal finding.

This follow-up keeps the 400 Pa response and its existing saturated-hot-zone,
PI-window, running-cooling-handler and mixed-air eligibility gates. Its one-kelvin
smooth pressure ramp, one-hour prestart, V4 PI gains, bands, water/pump commands
and 288-step cooling hold remain unchanged.

In the shipped VAV model, Q_i is proportional to d_i*sqrt(P). When pressure
increases, leave eligible hot targets' damper commands intact. For every other
zone on that handler, project its original command to its effective opening
(including the 0.1 ventilation floor), then request:

    d_new = max(0.1, d_old * sqrt(P_old_effective / P_new_effective))

Effective pressure includes the model's running floor. The ratio is computed
from the shipped zero-based pressure normalization. This preserves nominal
non-target airflow where the ventilation floor does not bind; it cannot
preserve it below that floor. It does not guarantee unchanged thermal output,
since SAT/return temperatures and thermal feedback are still shared.

| Arm | Pressure response | Additional guard |
|---|---|---|
| cfb_ctrl | Disabled | Exact V4 control |
| cfb_comp | 400 Pa ceiling + non-target flow compensation | None beyond existing eligibility |
| cfb_guard | Same compensation | Veto if a near-cold non-target needs an opening below the floor |

Near-cold means temperature at or below the lower comfort bound plus 0.5 K.
A missing non-target damper index also vetoes the handler modification. All
updates are collected before writing, so a veto cannot leave partially changed
dampers. Reheat valves are not scaled. The guard reads delayed observations
and is not a physical safety guarantee. Constants and gates are fixed before
new outcomes; no model parameter uses building/weather identifiers.

All measured episodes have 8,640 steps. The predeclared protocol is:

1. Reproduce e_sync on training building_train_7/phoenix_3 within 1e-10 mean
   reward using the disabled control.
2. Both candidates get training building_train_1/fairbanks_1,
   building_train_7/phoenix_3, building_train_4/atlanta_5,
   and building_train_7/denver_5.
3. Every training-passing arm gets dev building_9002/phoenix_2,
   building_9002/atlanta_4, building_9002/singapore_2,
   and building_9001/fairbanks_3.
4. Gates require a lower matched mean and no individual case more than 10%
   worse. The lowest passing dev mean selects full training-30/dev-20 coverage,
   reusing the selected candidate's completed episodes.
5. Only passing both full cohorts permits standalone notebook packaging,
   real-observation action-parity verification and a submission description.
   The V4 artifact remains intact and no Kaggle submission is sent automatically.

The old uncompensated 400 Pa results are retained as an ablation comparator on
matching cases; their file hash is captured in the new protocol. They are not
additional independent evidence. These deliberately targeted cases and
repeatedly used dev buildings support exploratory development, not untouched
confirmation. The test split stays unused.

The scheduler retains RAM gates and 20-second launch spacing, saves every
completed rollout, and records source/config/reference hashes and attempts.
Check existing processes before resuming:

```powershell
..\.venv\Scripts\python.exe -u -m hvaclab.flow_balance_experiment
```

Progress: `experiments/runs/flow_balance/status.json`.
Report: `reports/FLOW_BALANCE_REVIEW.md`.
Paper: a separate `reports/paper/flow_balance_results.tex` table appended to
the existing manuscript; all old data remain retained. A desktop completion/
failure watcher is active.
