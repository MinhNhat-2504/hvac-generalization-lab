# Cooling hold outside the PI window

This experiment follows the failed delay-compensation dev screen. It retains
`e_sync`'s flat temperature band and zone PI, heating pressure clamp, reheat
airflow override, and hot-water coordination. Only the timer that repeats a
previously requested cooling run changes outside the PI window. The PI window
itself includes the one-hour preconditioning lead-in.

| Variant | Daytime hold | Maximum hold outside PI window |
|---|---:|---:|
| s_hold_ctrl | 288 steps | 288 steps (unchanged e_sync) |
| s_night0 | 288 steps | 0 steps |
| s_night12 | 288 steps | 12 steps (60 minutes) |

Each step is 300 seconds: 288 steps equal 24 hours. An earlier prose
description incorrectly called this 12 hours; no parameter or result changed.

An actual heating or cooling request still runs the plant at night. This is
not a thermostat setback and does not forcibly switch off necessary nighttime
conditioning. Shorter persistence could save unoccupied fan/chiller cost, but
could also lose useful preconditioning or increase cycling. Performance is
an empirical question, not guaranteed by the code change.

The reference is completed `e_sync` coverage: training 30 episodes at
0.014453943529822781, dev 20 episodes at 0.016684401534296, and the
participant-provided Kaggle V4 public score is 0.01984. New local scores are
not mixed with the Kaggle public score.

Before outcomes, the screening pairs and gates are fixed to the same sets
used for the delay-compensation experiment:

1. One full unchanged-control episode on training building_train_7/fairbanks_7.
   Its mean reward must match e_sync within 1e-10.
2. Both candidates on training building_train_1/fairbanks_1,
   building_train_7/fairbanks_7, and building_train_8/phoenix_4.
3. Only the lowest-mean passing training candidate advances to dev
   building_9001/fairbanks_3, building_9000/denver_1,
   building_9002/phoenix_2, and building_9003/singapore_2.
4. Training and dev gates require a lower matched mean and no case more
   than 10% worse. A passing dev candidate advances to all 30 training and
   20 dev episodes, reusing its completed screen episodes. The test split
   is untouched. All comparisons require 8,640 steps per episode.
5. Only if full coverage also passes both gates does the pipeline build a
   separate notebook and check action parity between the standalone and
   repository policies on 576 actual-observation steps in each of two
   training scenarios. It then records the notebook path and writes a
   suggested submission description. Nothing is submitted to Kaggle.

Per-job results, attempts, source/config/reference hashes, and current state
are under `experiments/runs/hold_schedule/`. Check for an existing scheduler
before resuming with `python -m hvaclab.hold_experiment`.

The runner uses at most two isolated interpreters, checks RAM before starting
a second, saves every completed job, and updates `reports/HOLD_SCHEDULE_REVIEW.md`.
It maintains a separate `reports/paper/hold_results.tex` table and rebuilds the
paper PDF at completed stages. Earlier delay-compensation results, the older
paper body, all historical statistics, and the V4 submission artifact are
preserved. A desktop notification watcher is active for this run.
