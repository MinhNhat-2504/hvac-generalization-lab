# Two fixed follow-ups to submitted Kaggle Version 9

2026-11-02. User confirms Version 8 (`persist_432`) and Version 9
(`damper11_less`) are running/scoring on Kaggle. No organizer scores for them
have been provided. Version 7 public 0.01922 remains best reported.

Completed prior pipeline: hold-release 61 episodes rejected; valve refinement
61 rejected versus persist_432; heating-damper study 91 selected damper11_less.
Selected local training mean 0.0140406845 (-2.254% vs V7, -0.988% vs V8), dev
0.0162456810 (-1.244% vs V7, -0.04786% vs V8). The very small dev advantage
does not justify confidence about the organizer ranking. Preserve all data.

## Predeclared queue

Both new studies use the V9 controller: global reheat h**.75, cooling hold
432 steps, heating-only damper h**1.25. Same PI state/gains, schedules, air/water
coordination and outside-PI commands. Do not change by building/weather label.

| Study | Candidate changes | Identity control |
|---|---|---|
| damper_refinement | Heating damper exponents 1.1, 1.4, 1.75 | Exponent 1.25 |
| damper_persistence | Hold 288, 576, 720 steps (24,48,60h) | Hold432 (36h) |

The first is a fixed neighborhood and stronger-response probe around the
measured heating-airflow curve; no smooth or monotonic score response is
assumed. The second tests whether the hold optimum changes after reducing
intermediate heating airflow. Hold288/576 were tested on the earlier damper
law, so these are interaction follow-ups rather than untouched hypotheses.

Every heating curve retains zero/full damper authority and cooling dampers;
all valve commands stay h**.75. Every hold uses the inherited refresh-on-cooling
and cancel-on-heating logic. A hold duration is persistence after the latest
cooling request, not a cap on total runtime. Changing airflow may hurt comfort,
and longer persistence may waste energy; neither direction is presumed useful.

## Execution and strict comparison

Existing pipeline dependencies are complete. Start damper_refinement first;
damper_persistence waits for it to complete and its scheduler to exit. Maximum
three active rollout workers total, resource thresholds unchanged. Both queues
use independent supervision, exit logs, input hashes and checkpoint recovery.
Launch: `reports/scripts/start_v9_followups.ps1`. Registration and assembly
are child-local, preserving submitted artifacts and shared registry files.

For EACH study: one fresh 8,640-step V9 identity control must match its cached
mean reward within 1e-10; then all three arms receive all 20 dev scenarios.
Rank arms whose complete mean beats V7, V4, V8, V9 and every other selected
full-cohort predecessor. Eligible arms receive all 30 training scenarios in
that order. Package only the first arm improving BOTH means against EVERY
comparator by more than 1e-10, then standalone verification/source marker.
Keep negative outcomes and move to the next candidate or queued study.

At most 151 full episodes per study before verification, 302 for both. No
guaranteed notebook: all candidates may fail. Report individual regressions,
cost components and exact comparator means. Dev is repeatedly tuned, training
is published consistency data; test split remains unused. Only Kaggle can
establish a public score below 0.01922 or below newly reported V8/V9 scores.

The second study retains fixed V9 settings rather than automatically combining
the first winner; it must still outperform that winner on both local cohorts.
Paper sections and separate result tables are appended prospectively and
updated at checkpoints/stage changes. All earlier results remain intact.
