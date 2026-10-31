# Conditional reheat-curve follow-up

Prospective protocol, 2026-10-31. The user has uploaded `valve8_gentle` as
Kaggle notebook Version 7; its organizer score is pending. Internal study
labels are not Kaggle notebook version numbers: the earlier rejected local
`geo7_*` study is a different experiment. V4 remains the best known organizer
score, 0.01984, until a better result is reported.

## Evidence and hypothesis

The global curve h**0.75 completed 91 full episodes, with training mean
0.0143645138 (-0.619% versus V4) and dev mean 0.0164503660 (-1.403%). Energy
and carbon savings outweigh small increases in average comfort penalty.
Training improves 14 cases, regresses 15 and ties one; dev improves 12 and
regresses eight. Full-cohort gains do not imply every scenario improves.

By weather group, training Fairbanks improves 2.742% while Phoenix worsens
1.122%; dev Phoenix instead improves 3.004%. These conflicting associations
motivate observable operating-condition gates, not a climate-specific rule.
Do not select parameters by weather/building label or infer causality from
these group averages.

Hypothesis: boost valves only when needed instead of boosting all active-PI
heating demands, reducing unnecessary reheat around the comfort band while
retaining useful extra heat delivery. Let h=max(0,-u) for the unchanged V4
zone PI output, and opening = h + w*(h**0.75-h).

| Key | Weight w |
|---|---|
| valve9_ctrl | 0 (exact V4 control) |
| valve9_curve_ctrl | 1 (global curve reproduction) |
| valve9_heat | 1 only when the final same-step AHU command is running and normalized SAT > 0 |
| valve9_cold | clip((comfort_low - observed_zone_temp)/0.5 K, 0, 1) |
| valve9_heat_cold | Cold-zone weight only when the AHU command condition also holds |

The command sign is an observable control-mode proxy, not actual supply-air
temperature or proof that the AHU heats; the AHU cannot add heat. Apply the
gate after parent cooling hold and heat-delivery coordination so it uses the
final same-step command. The cold blend is zero at/above the low band edge and
full at 0.5 K below it. The inherited temperature-observation delay remains;
there is no prediction or access to simulator state/reward in the policy.

PI gains, signed demand, integration rule/state, damper commands, water/pump,
AHU controls, schedules and cooling hold remain V4. Changes in temperatures
can affect future feedback; fixed-observation action invariance is not a
claim that closed-loop commands will be identical. All curves retain zero and
full valve authority. Outside active PI, inherit V4 unchanged. No geometry.

## Fixed measurement and selection

1. On building_train_1/fairbanks_1, run two complete fresh controls (8,640 steps
   each). Require the off gate to reproduce V4 and the all gate to reproduce
   valve8_gentle within 1e-10 mean reward. This cold case exercises heating.
2. Run all three fixed gates on all 20 dev scenarios (60 full episodes).
3. Rank candidates whose full-dev mean beats BOTH V4 and the submitted global
   curve by more than 1e-10. Evaluate all 30 training scenarios in that order,
   continuing after any failed training check.
4. Package the first candidate whose full-training mean also beats BOTH
   references. Verify standalone action parity, hashes and runtime source
   marker. Otherwise preserve the existing submissions and report failure.

At most 152 new full episodes before standalone checks. Compare complete,
equal-weight matched cohorts. Report components, regression counts and largest
increases; no arbitrary worst-case veto. Dev has been repeatedly tuned and
training is published consistency data. Neither is untouched validation; the
separate local test split remains unused. A local pass permits an experimental
submission, not a claim to beat V4's organizer score.

## Execution and preservation

Run `python -m hvaclab.gated_reheat_calibration`; results under
`experiments/runs/gated_reheat_calibration`. Freeze source/config/reference
hashes and dependency versions; use duplicate-run locking, per-episode atomic
records and matching-input resume. Maximum three workers, 20-second launch
spacing, 1.75 GiB available RAM and 1 GiB free disk before launches.
Desktop completion notice is supported; no proactive chat message is promised.
Build paper separately and replace the PDF only on success. Append a new
section/table and preserve all previous data and submitted notebook artifacts.
