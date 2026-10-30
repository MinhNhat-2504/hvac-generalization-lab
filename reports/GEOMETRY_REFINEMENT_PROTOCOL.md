# V6: refine geometry sensitivity against V5 and V4

V5 (`geo_pi_sqrt`) completed 30 training and 20 dev episodes: respective means
0.0141571250 and 0.0160920115, reductions of 2.05% and 3.55% versus V4.
Its standalone action parity passed. V5 is submitted; its organizer score is
still pending as of this protocol. V4 public score remains 0.01984.

Hypothesis: V5's successful geometry sensitivity may be too weak or too strong.
Its exponent 0.5 was a fixed first probe, not an optimized physical constant.
The full-dev weighted comfort penalty fell from 0.0034191613 (V4) to
0.0027456800 (V5), approximately 19.70%, while weighted energy increased about
0.70% and carbon increased about 0.53%. This motivates refining zone response
before altering plant coordination again. It does not prove a causal model.

## Fixed candidates and unchanged settings

For room area A, s = clip((A/2484)^p, 0.5, 2), Kp = 0.1s, Ki = 0.005s,
and raw integral limit = 200/s. Integral contribution remains bounded by one.

| Key | p | Purpose |
|---|---:|---|
| geo6_ctrl | 0.5 | Exact V5 reproduction |
| geo6_quarter | 0.25 | Weaker geometry dependence |
| geo6_threequarter | 0.75 | Moderately stronger dependence |
| geo6_linear | 1.0 | Linear area dependence before clipping |

Use the unchanged geometry implementation and training-only reference area.
All V4/V5 plant coordination, AHU temperatures/pressures, water control,
comfort bands, one-hour prestart, 288-step cooling hold, integration condition,
and activation schedule remain identical. No building or weather ID selects
parameters. Geometry mapping limitations and fallback remain those documented
in `GEOMETRY_CALIBRATION_PROTOCOL.md`.

## Evaluation and packaging

1. Require one new complete 8,640-step control episode to reproduce V5 within
   1e-10 mean reward. Baselines are 50 complete V5 episodes with unique scenario
   identities and matched complete V4 episodes; reject missing/invalid records.
2. Evaluate all three fixed candidates on all 20 dev scenarios (60 episodes).
3. Rank candidates whose full-dev mean beats BOTH V5 and V4 by more than 1e-10.
   Run all 30 training episodes for eligible candidates in dev-rank order,
   continuing after a training failure.
4. Package only the first candidate whose full training mean also beats BOTH
   references. Verify standalone action parity, preserve V4/V5 artifacts, and
   save notebook hashes, measured comparisons and a submission description.
   Otherwise record the negative result and retain V5; do not invent a V6 file.

Maximum 151 full episodes, plus standalone verification if selected. Individual
regressions are reported, without a hard per-scenario veto. Dev is repeatedly
used tuning data. Training is a published consistency check, not untouched
validation. The separate test split remains unused. Kaggle must establish any
organizer-score improvement; local means are not organizer-score forecasts.

The new run directory is `experiments/runs/geometry_refinement`. Freeze source,
configuration, geometry-map, V4/V5 reference hashes and installed dependencies.
Use the existing duplicate-run lock, atomic per-episode checkpoints, at most
three workers, 20-second launch spacing, and 1.75 GiB available-memory launch
threshold. Resume only matching frozen inputs. Results are appended as a new
paper section/table; the earlier measurements and artifacts remain intact.
