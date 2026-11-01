# Two queued follow-ups to Kaggle Version 8

Prepared 2026-11-02. User confirms `persist_432` is running on Kaggle as
notebook Version 8. Its organizer score is pending. Version 7 / valve8_gentle
remains best reported public score, 0.01922. Internal experiment labels are
independent of Kaggle notebook version numbers.

The persistence study completed 91 full episodes, selecting hold 432 steps
(36h). Training mean 0.0141808105 (-1.2789% versus V7), dev mean 0.0162534593
(-1.1970%). Training improves 18 cases, regresses nine, ties three; dev improves
12, regresses five, ties three. Action parity passed 1,152 observation pairs.
Local improvement does not establish the organizer score.

## Queue order and fixed hypotheses

The currently running sustained-cold hold-release study retains simulation
priority. Two independent queues follow it in a strict dependency chain:

| Order | Directory / module | Fixed candidate changes on persist_432 |
|---|---|---|
| Current | hold_release_calibration | Previously declared stale-hold release rules |
| Next | curve_refinement | Reheat h**p with p=.5,.625,.875 versus p=.75 |
| Then | heating_damper | Heating-only damper h**q with q=.75,1.25,1.5 versus q=1 |

Both new studies use the organizer-unconfirmed persist_432 base: hold 432,
global reheat exponent .75 unless it is the explicitly varied coefficient,
original V7 PI/schedules/AHU/water coordination. They do not automatically
combine earlier selected changes. Instead every selected preceding winner is
a required score comparator, so a new arm must outperform them on both cohorts.

Curve refinement tests whether the gain observed for h**.75 persists or can
improve with a 36-hour hold. The p=.5 curve was previously tested on 24-hour
hold; this is an interaction follow-up, not a new untouched hypothesis.
The .625 and .875 neighbors are fixed before these new rollouts.

Heating-damper refinement keeps the valve h**.75. For signed PI demand u<0,
damper becomes (-u)**q. For u>=0 and outside active PI, damper remains the
parent command. The PI state/demand and conditional integration do not change;
zero/full actuator authority remain. Higher q reduces intermediate heating
airflow, lower q increases it. Airflow both transports reheat and admits intake
load, so either direction can hurt comfort or energy. Fixed-observation action
invariance does not mean unchanged future closed-loop trajectories.
No building/weather label, map, reward or true simulator state is read.

## Measurement and gates

Each study waits for all declared predecessors to reach status complete AND
their schedulers to exit. Failure or missing status does not trigger the next.
It then freezes source/config/reference hashes and installed dependencies,
including every selected preceding winner's complete results.

1. Fresh 8,640-step identity control must match persist_432 within 1e-10 reward.
2. Run all three arms on all 20 dev scenarios.
3. Rank candidates whose full-dev mean beats V7, V4 and every selected prior
   full-cohort winner. Evaluate all 30 training scenarios in that order,
   continuing after a training failure.
4. Package the first candidate whose complete training mean also beats every
   comparator by more than 1e-10. Verify standalone parity and source marker.
   Otherwise record failure and preserve earlier artifacts.

Maximum 151 full episodes per study before standalone checks, so at most 302
for these two queued studies. Dev has been repeatedly tuned; training is a
published consistency check. Test split remains unused, individual regressions
are retained, and only Kaggle establishes a score below .01922.

## Execution and paper updates

Launch both independent waiting queues with:

```powershell
powershell -NoProfile -File reports/scripts/start_research_pipeline.ps1
```

`hvaclab.pipeline_queue` supervises each allowlisted study. Candidate registration
and standalone adaptation occur only in child memory via `pipeline_entry`;
the active study's shared registry/builder/source files are untouched. Per-run
locking prevents duplicate queues. Maximum three rollout workers total because
the studies are sequential; 1.75 GiB available RAM, 1 GiB disk, 20s launch spacing.
The independent supervisor logs exits, waits for orphan workers and retries
from valid checkpoints. Restart queue launchers after reboot/logoff.

Each checkpoint writes a distinct report and paper table, including completed
full-cohort scores, stage and decisions. Stage changes rebuild the PDF in an
isolated directory and replace it only on success. Earlier paper tables,
measurements and submitted notebooks are retained. Two queued hypotheses are
prepared now; replenish the queue with evidence-based hypotheses after results,
rather than generating unbounded untested variants automatically.
