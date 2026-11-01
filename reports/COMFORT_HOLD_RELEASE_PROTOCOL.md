# Prepared next hypothesis: release stale cooling persistence under sustained cold

2026-11-02. Best reported organizer result remains Kaggle V7 / valve8_gentle,
0.01922. This is a prepared controller and prospective protocol, not measured
performance and not a packaged notebook. The fixed-duration persistence study
is still validating training and has priority for simulation RAM.

## Rule and scope

Extend the unchanged V7 valve curve and 288-step hold. During the PI window,
for each AHU require all finite observed zone temperatures <= comfort-band
centre AND at least one <= comfort low minus a fixed cold deficit. Require
the condition on consecutive observations, resetting on a warm/inactive or
invalid observation. On confirmation, clear only that handler's inherited
hold timer and stored cooling command BEFORE calling the parent controller.

Do not turn the AHU off directly or rewrite supply temperature/pressure,
recirculation, dampers, valves or pump commands. The parent can issue a new
cooling request (which rearms hold), heating request or idle; existing heat
airflow/water coordination remains in charge. Global reheat exponent .75,
PI gains and conditional integration, bands and schedule remain V7.

Hypothesis: the last cooling command can persist after every zone is below
the band centre while a cold zone incurs loss; clearing stale memory can
reduce overcooling and counteracting reheat. This mechanism may be infrequent
because normal aggregate heating already cancels hold. Therefore no gain is
assumed. An unreachable cold outlier may also clear useful cooling reserve;
the all-zone centre requirement and consecutive samples are hypotheses, not
safety or comfort guarantees. The inherited observation delay remains.
No building/weather label, geometry, reward or true simulator state is used.

## Fixed prospective arms

| Arm | Cold deficit | Consecutive steps |
|---|---:|---:|
| Disabled control | Disabled | 3 |
| Conservative | 0.5 K | 3 (15 min) |
| Earlier | 0.25 K | 3 (15 min) |
| Longer confirmation | 0.5 K | 6 (30 min) |

After the current study finishes, integrate these keys into the registry and
standalone builder under a NEW frozen run directory. Never alter the running
persistence study dependencies. First run one fresh full disabled control and
require V7 reproduction within 1e-10 mean reward. Evaluate all three arms on
all 20 dev scenarios, then all 30 training cases for eligible arms in full-dev
rank order, continuing after a training failure. Compare with V7, V4 and any
new full-cohort winner from the current persistence study; require both full
means to improve against every such reference before packaging. Maximum 151
full episodes before standalone checks. Do not select using a partial cohort.

Record hold release counts so a null result can distinguish inactive mechanism
from ineffective intervention. Run standalone/notebook-wrapper parity before
packaging. Preserve existing artifacts and append paper results; test split
unused. Repeated dev tuning remains exploratory and only Kaggle can establish
a score below .01922.

## Current implementation status

`hvaclab/policies/comfort_hold_release.py` implements the isolated policy;
`tests/test_comfort_hold_release.py` checks disabled identity, consecutive-step
release, warm reset, nonfinite/inactive/hot-reserve veto and preservation of
new parent cooling/heating requests. No registry/builder or existing scheduler
was changed to introduce this candidate, and no performance rollout has started.

## Automatic sequential queue

User authorized automatic execution after persistence validation. The detached
`hvaclab.hold_release_queue` waits for status complete and scheduler exit, then
prepares the next frozen protocol and independently supervises the release
study. A preceding failure does not trigger the next study. Child-local
registration/assembly in `hold_release_entry` leaves shared registry and builder
unchanged. Compare also with any selected full-cohort persistence winner.
Launch with `reports/scripts/start_hold_release_queue.ps1`. State/logs live in
`experiments/runs/hold_release_calibration`; ordinary resource checks, locking,
orphan-worker wait and bounded retries apply. No concurrent rollout is opened.
Reboot/logoff still requires restarting the queue launcher.
