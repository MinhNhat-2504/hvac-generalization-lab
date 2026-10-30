# Earlier PI preconditioning — 28 October 2026

This is a new experiment on the unchanged e_sync controller, following failed
pump and shortened-hold screens. Only `prestart_hours` changes: explicit
one-hour control, two-hour candidate, and three-hour candidate. Bands, PI gains,
cooling hold, fan pressure, water reset and pump requests retain V4 settings.
Different prestart histories can change later daytime actions.

Before candidate evaluation, run a complete one-hour-control diagnostic on
training building_train_7/phoenix_3, currently the largest training comfort
contributor under e_sync. Group actual reward terms by their start timestamps
in the scenario schedule's timezone. Check whether the first two scheduled
comfort hours account for at least 25% of the full episode's comfort loss.
This concentration threshold is fixed before inspecting the diagnostic result;
it is a heuristic support check, not a statistical significance test. If the
share is below 25%, stop this proposed prestart sweep and record that the
diagnosis did not support its premise, without claiming a control improvement.
The diagnostic must reproduce the cached full e_sync score within 1e-10.

If supported, evaluate both earlier starts on these complete training cases:

| Building | Weather | Role |
|---|---|---|
| building_train_1 | fairbanks_1 | Deep-cold energy/comfort control |
| building_train_7 | phoenix_3 | Largest residual training comfort contribution |
| building_train_4 | atlanta_5 | Second-largest residual training comfort contribution |
| building_train_7 | denver_5 | Third-largest residual training comfort contribution |

Every training-passing arm advances to a development screen of
building_9002/phoenix_2, building_9002/atlanta_4,
building_9002/singapore_2, and building_9001/fairbanks_3. The first three
target the largest residual dev comfort contributors; the last is a heating
control. Selection uses previously observed local baseline loss, so these
screens are deliberately targeted and cannot be treated as representative
global improvement estimates or untouched validation.

Each performance gate requires a lower matched mean and no individual case
more than 10% worse, with 8,640 steps per episode. Select the lowest passing
dev mean, then evaluate all 30 training and 20 dev cases, reusing its completed
episodes. Only passing both complete cohorts permits verified standalone
notebook packaging and a suggested submission description. The existing V4
artifact is retained, and no Kaggle submission occurs automatically. Repeated
dev use is exploratory model selection; the test split remains unused.

The complete diagnostic is reused as the control, not counted twice as
independent evidence. Its source/config hashes, result fields, reference
agreement and file hash are checked and recorded. Results and failed gates
are added to a separate report and paper table, preserving previous data.

The scheduler retains the 1.25/1.5 GiB available-RAM gates and 20-second launch
spacing. Planned progress: `experiments/runs/prestart/status.json`.

Diagnostic correction before candidate evaluation: the initial profile converted
protobuf timestamps to US/Pacific, but these scenarios run on naive simulation
wall-clock timestamps used directly by the observation clock and comfort
schedule. That conversion shifted the hour bins by seven hours. The original
`prestart_profile.json` is retained for audit. The corrected full episode in
`prestart_profile_policyclock.json` groups on the observed policy clock,
reproduces the same reference score, and additionally splits actual reward
comfort loss by cold/hot direction. Candidate gating uses the corrected
diagnostic only. No simulator or controller clock was modified.
