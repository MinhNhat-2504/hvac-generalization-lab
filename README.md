# hvac-generalization-lab

Transferable HVAC control policies for unseen buildings. Entry for the
[NeurIPS 2026 Smart Buildings competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/)
(Kaggle × NeurIPS 2026, hosted by University of Washington with the NSF AI
Institute in Dynamic Systems, Harvard, Columbia and Kajima).

The task is one control policy that runs the HVAC plant of commercial
buildings it has never seen — different floor plans, zone counts, air-handler
layouts and climates — trading occupant comfort against energy cost and carbon
over a 30-day, 8,640-step episode. Scoring is the mean over 30 held-out
scenarios of the absolute mean per-step reward. **Lower is better.**

```
score = mean over scenarios of | mean over steps of r_t |
r_t   = 0.6 * comfort  -  0.2 * energy_cost  -  0.2 * carbon      (all penalties)
```

## The finding this entry is built on

Update, 2026-10-29: V4/e_sync remains the submitted reference (public score
**0.01984**). The 251-episode scalar PI study found no candidate improving
both complete local cohort means. The next bounded study tests zone gains
scaled by participant-visible room area; see the
[geometry PI protocol](reports/GEOMETRY_CALIBRATION_PROTOCOL.md) and
[live results](reports/GEOMETRY_CALIBRATION_REVIEW.md).
Paper additions preserve the earlier numerical results.

Follow-up: V5 (`geo_pi_sqrt`) passed full local training/dev checks (mean
reductions 2.05%/3.55% versus V4), passed standalone parity and was submitted;
its Kaggle score is pending. The prospective
[V6 refinement](reports/GEOMETRY_REFINEMENT_PROTOCOL.md) tests three geometry
exponents and requires both full cohort means to beat **V5 and V4** before
packaging. See its [results](reports/GEOMETRY_REFINEMENT_REVIEW.md).

Organizer feedback, 2026-10-30: the user's Kaggle screenshot reports V5 at
**0.01996**, versus V4 **0.01984** (V5 is 0.60% worse). V4 remains better on
Kaggle despite V5's local gains. See the [recorded feedback](reports/V5_KAGGLE_FEEDBACK_2026_10_30.md).

V6 (`geo6_threequarter`) completed local validation and standalone parity and
was submitted; organizer score pending. The prospective
[V7 split-gain study](reports/SPLIT_GEOMETRY_CALIBRATION_PROTOCOL.md) separates
P/I geometry sensitivity and requires complete training/dev means below V6,
V5 and V4 before packaging. Its [results](reports/SPLIT_GEOMETRY_CALIBRATION_REVIEW.md)
are separate from the retained earlier measurements.

Later organizer feedback reports V6 **0.03833**: V4 **0.01984** remains best.
The [feedback and local notebook audit](reports/V6_KAGGLE_FEEDBACK_2026_10_30.md)
retain the local results while distinguishing them from the organizer outcome.

2026-10-31: V7 completed 91 episodes and failed training; no notebook was
packaged. The next [reheat-curve study](reports/REHEAT_CURVE_PROTOCOL.md) returns
to V4 and adjusts only active-PI valve response, without geometry scaling.
It requires lower complete training/dev means than V4 before packaging and
adds a runtime variant/source marker for Kaggle log checks. See the
[study results](reports/REHEAT_CURVE_REVIEW.md); a local pass cannot establish
an organizer improvement.

The user has submitted `valve8_gentle` as Kaggle notebook **Version 7**;
organizer score pending. Internal study labels differ from Kaggle notebook
versions. The next [conditional reheat study](reports/GATED_REHEAT_PROTOCOL.md)
tests command-mode and cold-zone gates, and requires complete local means
below both V4 and the submitted global curve. See its
[recorded results](reports/GATED_REHEAT_REVIEW.md). Earlier results are retained.

Independent [background supervision](reports/STUDY_SUPERVISOR.md) can adopt the
running gated-reheat scheduler, record exits and restart from saved episodes.
Launch with `powershell -NoProfile -File reports/scripts/start_study_supervisor.ps1`.
It does not register Windows startup or prevent sleep.

2026-11-01: Kaggle Version 7 / `valve8_gentle` is the best reported public
score, **0.01922**. Conditional reheat gates completed 62 episodes and all
failed dev versus Version 7; no replacement notebook was produced.
The next [persistence study](reports/PERSISTENCE_PROTOCOL.md) tests 12/36/48-hour
cooling holds against Version 7's 24 hours while retaining its valve curve.
See [results](reports/PERSISTENCE_REVIEW.md). Earlier measurements stay retained.

2026-11-02: `persist_432` passed local training/dev (-1.28%/-1.20% versus
Version 7) and standalone parity; user is running it on Kaggle as **Version 8**,
score pending. The [research pipeline](reports/RESEARCH_PIPELINE_PROTOCOL.md)
keeps the active hold-release study followed by two prepared sequential
studies: reheat-exponent refinement, then heating-damper response. Each must
beat all selected preceding full-cohort winners before packaging. Separate
paper tables and reports update during execution; existing data remain retained.

Later on 2026-11-02, the queue completed: hold-release and valve refinement
were rejected; heating-damper exponent 1.25 passed both complete cohorts and
standalone parity as `damper11_less`. The user is running it on Kaggle as
**Version 9**, alongside Version 8; both scores pending. The
[two V9 follow-ups](reports/V9_FOLLOWUPS_PROTOCOL.md) refine its heating-damper
exponent, then hold-duration interaction, with strict comparison against all
selected earlier winners. Version 7 remains best reported public 0.01922.

The comfort term is **occupancy-weighted**, and the weighting is applied as a
hard gate. From `smart_control/reward/base_setpoint_energy_carbon_reward.py`:

```python
if total_occupancy > 0.0:
    normalized_productivity_regret = actual/max - 1.0
else:
    normalized_productivity_regret = 0.0
```

So on any step where the building is empty, the comfort term is **identically
zero regardless of zone temperature**, and the step's reward is nothing but
`-0.2*energy - 0.2*carbon`.

Occupants arrive between hour 6 and 12, leave between 13 and 19, and
`is_work_day` forces every occupant away on weekends and holidays. Over a
30-day strip that leaves **about 60% of the 8,640 scored steps with nobody in
the building**. Every joule the plant burns in those steps buys nothing.

The four shipped baselines all hold a setback band around the clock, so they
all pay that bill. Measured on `building_train_1/fairbanks_1`, the scenario
where they do worst, the weighted split of `pid_water`'s loss is

| term | contribution |
|---|---|
| comfort | 0.087 |
| energy  | 0.077 |
| carbon  | 0.122 |

**69% of the loss is energy and carbon**, and the largest single term is carbon
from the gas boiler — holding 289 K against −42 K ambient all night, for no
comfort credit.

### The other half: comfort is worth far more than energy while occupied

The regret is a weighted *mean* of per-zone shortfall, and shortfall is
`deviation / 5 K` inside the knee. Every occupied zone sitting 1 K outside its
band therefore costs `0.6 × 0.2 = 0.12` reward per step. One kWh of
electricity costs `0.2*(0.16/6.37) + 0.2*(0.103/15.14) = 0.0064`.

Closing 1 K of building-wide deviation is worth **≈19 kWh per five-minute
step — 225 kW held continuously**, against a 320 kW plant. While the building
is occupied, energy is nearly free next to comfort. While it is empty, comfort
is free and energy is the entire bill.

That asymmetry is the whole strategy: **spend freely when watched, spend
nothing when not**, and treat the boundary between the two as a deadline
problem.

## The obvious move, and why it failed

Switching the plant off whenever the building is empty
(`hvaclab/policies/occupancy_gated.py`) measured **worse** than `pid_water` on
all three reference scenarios. Two independent reasons, neither visible from an
aggregate score:

**1. Being empty inside the scored window is not an opportunity.** Arrivals are
independent stochastic draws from hour 6, so the building is routinely empty
for a few steps *after* the window opens — and regret is normalized by *total*
occupancy, so the first person through the door carries their zone's entire
comfort score alone. Switching off at 06:05 because nobody has arrived yet
means being off, with every zone out of band, when they do.

**2. Driving to the comfort band outside the window costs more than coasting
saves.** The first attempt aimed at the comfort band (294–297 K) whenever it
judged recovery necessary, where the baselines use the wide eco band
(289–298 K). In Fairbanks that is 5 K of extra heating all night, and it showed
up exactly where it should — unoccupied-step loss went **up**, 447 → 830,
against a comfort gain of only 0.010.

So the lever is not an on/off gate. It is the **band itself**.

## Method

**`hvaclab/policies/deadline_band.py`** — no modes. One quantity per step, the
band to hold now, handed to the inherited control law:

| when | band |
|---|---|
| inside the scored window | the comfort band, exactly |
| outside it, deadline far | comfort band widened by the full setback |
| outside it, approaching | widened by a fraction, ramping linearly to the comfort band at the deadline |

The inherited law already does the right thing with a wide band: when every
zone is inside it, `pid` idles its air handler, shuts the damper, closes the
reheat valve, drops the water loop to its minimum and clears every run
command. **A band wide enough to contain the building _is_ the off action**, and
it is reached through the same code path as everything else, so there is no
separate shutdown branch to get wrong.

Setback depth is a measured parameter, not a feasibility calculation. The first
attempt derived it from an online recovery-rate estimate, and that estimate was
the weak link — taken from the *worst zone's* deviation closing, it overstates
what a shared plant can do for a whole building at once. Bounded also looks
right on merit: `fixed_setpoint` runs one band around the clock with no setback
at all and **beats `pid_water` on `fairbanks_1`**, because recovering from a
setback runs the fan at full static pressure. Which leads to the second change.

**`hvaclab/policies/fan_economy.py`** — a direction-aware static pressure law.
Every policy in the `pid` chain sets `pressure = clip(2|u| - 1, -1, 1)`, scaling
the most expensive fuel *up* with demand. `abs(u)` does not distinguish the two
directions and they are not symmetric:

- **Heating.** The air handler cannot heat; all heat comes from the VAV reheat
  coils. In cold weather the supply-air setpoint is inert too — the handler
  delivers `min(mixed_air, setpoint)`, and with recirculation capped at 0.70,
  return near 293 K and ambient at 231 K, mixed air is 274 K, already below the
  285 K floor the setpoint can command. Raising static pressure pushes more
  274 K air past the coil, and every cubic metre has to be heated from 274 K
  before it helps the zone. **The fan work is spent, then gas is spent undoing
  the air it moved.** The zone's heat loss fixes the energy needed; it can
  arrive as a lot of slightly warm air or a little hot air, and the second is
  strictly cheaper because the gas is the same either way and the fan is not.
- **Cooling.** Air is the only delivery mechanism, so flow must be bought — but
  the trade still favours temperature over flow, and `pid` already drives the
  supply setpoint to its 285 K floor at full effort.

First measurement, one unoccupied day on `fairbanks_1` (a pure energy/carbon
comparison, no comfort at stake): **0.00908 → 0.00499**, energy −57%,
carbon −38%.

## Layout

```
smart_buildings/    official starter kit, Apache-2.0, left as upstream
hvaclab/            everything written for this entry
  policies/         controllers
  runner.py         parallel sweep over scenarios; the leaderboard metric
  sweep.py          CLI: python -m hvaclab.sweep --policies occ
  registry.py       policy lookup by key, starter-kit keys included
experiments/runs/   sweep results, tracked as the measurement record
reports/            notes and paper drafts
```

## Running it

The heating-airflow ablation completed 20 full episodes on four high-comfort-loss
scenarios plus a deep-cold control. `d_heat` reduced their mean score from
0.06122466 to 0.04162332 (32.02%), but worsened `fairbanks_1` by 30.31%.
The submission remains `h_288` pending broader validation. See the
[heating-mode review](reports/HEAT_DELIVERY_REVIEW.md) for the paired results,
the hot-water coordination issue, and reproduction commands.

The follow-up `e_sync` coordinates the water pump and inherited water reset
with reheat airflow. It scores **0.02252922 on the same five scenarios**
(63.20% below `h_288`, 45.87% below `d_heat`) and **0.01480846 on three dev
scenarios from one new building**, against `h_288`'s 0.04542558 on those exact
three. Singapore regresses by 1.09%; the heating scenarios drive the gain.
See the [coordination experiment](reports/HEAT_SYNC_REVIEW.md) for coverage
and limitations. Build its separate candidate notebook with:

```bash
python -m hvaclab.submission.experimental --variant e_sync --verify
```

A subsequent 1 K heating setback (`n_sync_s1`) failed its two-scenario screen:
mean score increased 1.40%, so it did not advance to dev or replace `e_sync`.
See the [night-setback result](reports/NIGHT_SETBACK_REVIEW.md).

The [source and paper audit](reports/SOURCE_PAPER_AUDIT.md) catalogs earlier
negative results, distinguishes the manuscript's older controller from Version 4,
and documents observation timing and sensor semantics to check before new tuning.

The [broad-coverage report](reports/E_SYNC_COVERAGE.md) tracks the remaining
25 training and 17 dev rollouts for `e_sync`. Its resumable runner saves each
completed episode separately and checks source/config/cache provenance before
reusing results. Current progress is in
`experiments/runs/e_sync_coverage/status.json`; the original coverage plan is
an immutable preparation snapshot. Partial averages are not full-set results.

```bash
# Resume only when the previous scheduler has stopped; at most two workers.
python -m hvaclab.coverage --workers 2

# Update the report while running; then collect/evaluate a separate shadow trace.
python reports/scripts/coverage_report.py --watch --shadow-after
```

The shadow experiment learns `obs[t+1] - obs[t]` from observations available at
time `t` and action `t-1`, reflecting the verified one-step temperature delay.
It evaluates each prediction before learning its target, compares against
persistence and constant-trend forecasts, and never changes controller actions
or reads true simulator temperatures. Forecast accuracy alone does not establish
a better controller. Its default trace is diagnostic, not a full-episode score.

The completed `e_sync` sweeps score **0.01445394 on training (30)** and
**0.01668440 on dev (20)**, reductions of 47.90% and 51.63% against matched
`h_288` references. The participant-provided Kaggle V4 public score is **0.01984**.
The [delay-compensation protocol](reports/DELAY_COMPENSATION_PROTOCOL.md) fixes
the next closed-loop screen and advancement gates before seeing its results.
[Progress and results](reports/DELAY_COMPENSATION_REVIEW.md) are saved per rollout;
the paper has an additive dated section and automatically maintained experiment
tables. The original source and PDF are preserved under `reports/paper/history/`.

Delay compensation finished with 11 valid full episodes: its training-selected
candidate regressed **2.98% on the four-case dev screen**, so it was rejected.
The [scheduled cooling-hold protocol](reports/HOLD_SCHEDULE_PROTOCOL.md) tests
shorter cooling-run persistence outside the PI window, preserving flat bands.
[Progress and results](reports/HOLD_SCHEDULE_REVIEW.md) are saved automatically.
Only a candidate passing complete matched training/dev coverage proceeds to
standalone notebook packaging and action-parity verification. The pipeline
records the resulting notebook path; it never submits to Kaggle.

The selected zero-hold candidate subsequently failed dev (+2.44% mean score),
so that experiment stopped without a new submission. A separately labeled
[one-hour follow-up](reports/HOLD60_FOLLOWUP_PROTOCOL.md) evaluates the remaining
`s_night12` candidate on dev with unchanged parameters. Its cached training
episodes are explicitly reused, not counted as new independent evidence.
[Follow-up results](reports/HOLD60_FOLLOWUP_REVIEW.md) and a new paper table
are maintained without replacing the previous experiment's measurements.

The one-hour follow-up also failed dev (+3.23%). The next fixed
[pump-pressure protocol](reports/GUARDED_PUMP_PROTOCOL.md) compares an uncapped
control, a simple cap after e_sync coordination, and a cap with a cold-zone/
valve-demand guard. [Progress and results](reports/GUARDED_PUMP_REVIEW.md) are
recorded separately. The scheduler waits for available RAM before starting a
simulation; `waiting_for_memory` distinguishes that state from active rollouts.
All previous experimental results and the V4 artifact remain retained.

The guarded-pump screen finished without a replacement: its guard improved
training by 0.42% but regressed dev by 0.37%. The next
[heating-command pump protocol](reports/HEATING_PUMP_PROTOCOL.md) preserves
reference pressure during any simultaneous cooling/neutral AHU request and
compares a stricter same-step reheat-airflow-override gate.
[Progress and results](reports/HEATING_PUMP_REVIEW.md) are maintained separately.
Passing complete matched coverage leads automatically to a verified standalone
notebook and suggested description; the V4 file is retained.

Both heating-pump arms subsequently failed dev (+0.45%, with identical scores
on the screened cases). A full residual-comfort diagnostic then rejected an
[earlier-prestart sweep](reports/PRESTART_PROTOCOL.md): only 12.00% of comfort
loss lay in the first two scheduled hours, below the predeclared 25% gate.
Use `hvaclab.comfort_profile` for clock-aligned profiling; the initial
timestamp-converted `morning_profile` result is retained as an audit artifact.
The [cooling-pressure protocol](reports/COOLING_PRESSURE_PROTOCOL.md) tests
bounded airflow support for hot zones with saturated PI cooling demand.
[Progress and results](reports/COOLING_PRESSURE_REVIEW.md) and a separate paper
table are maintained without replacing previous data.

The bounded-pressure arms failed dev (+3.42% for 400 Pa; +4.88% for 600 Pa).
The [flow-balance follow-up](reports/FLOW_BALANCE_PROTOCOL.md) compensates
non-target damper openings when raising shared pressure, with a second arm
protecting near-cold zones constrained by minimum ventilation. The previous
pressure-only results remain an archived comparator, not independent evidence.
[Progress and results](reports/FLOW_BALANCE_REVIEW.md) and an additional paper
table are maintained separately. Complete matched coverage and standalone
parity are still required before packaging a replacement candidate.

Flow compensation also failed its targeted dev screen (+2.13% unguarded,
+1.70% guarded). The next [bounded PI calibration](reports/PI_CALIBRATION_PROTOCOL.md)
keeps e_sync's plant coordination and preserves maximum integral contribution
while testing four one-coordinate gain neighbors on **all 20 dev scenarios**.
Dev-ranked candidates receive complete training-30 validation automatically;
if none beats both full means, a fixed second batch checks four joint corners.
[Progress and complete-cohort results](reports/PI_CALIBRATION_REVIEW.md) are
saved per rollout. Selection follows the mean-score objective; individual
regressions are reported rather than used as an additional hard cutoff.
A passing candidate gets a standalone notebook, validation record and suggested
submission description. V4 and every earlier measurement remain retained.

```bash
pip install -e .

# Rebuild the 30 shipped scenarios (not tracked: ~100 MB of regenerable rasters)
python -m smart_buildings.run_scenario_generator

# Full 30-scenario sweep, parallel. ~20 min on 14 workers.
python -m hvaclab.sweep --policies occ --out experiments/runs/occ.json

# One scenario with charts, via the starter kit's own evaluator
python -m smart_buildings.evaluate --policies pid_water \
    --buildings building_train_1 --weathers fairbanks_1
```

Always sweep at the default 8,640 steps. A shorter horizon truncates the
eco-setback hours and so flatters schedule-aware policies — this one most of
all, since that is where its advantage lives.

## Reference points

Leaderboard metric on the 30 shipped training scenarios, as recorded in
`baseline_returns.json` (lower is better):

| policy | score |
|---|---|
| `bang_bang` | 0.2252 |
| `pid` | 0.0674 |
| `fixed_setpoint` | 0.0659 |
| `pid_water` | 0.0541 |
| per-scenario oracle over the four | 0.0473 |

The oracle row is the floor for any method that just picks among the shipped
baselines, and it is only 12.7% below the best single one — so baseline
selection is not where the headroom is.

> ### ⚠ `baseline_returns.json` no longer matches the simulator
>
> The simulator is **exactly deterministic** — the same policy on the same
> scenario reproduces to ten decimal places, verified across two runs in one
> process. So any discrepancy is systematic, not noise.
>
> Running `pid_water` on `building_train_1/fairbanks_1` with the current code
> gives **0.21392**. `baseline_returns.json` records **0.21168** — 1.1% apart.
>
> The file was committed in `3bb6c32`, the initial starter-kit commit, and
> never regenerated. Commit `9567419` ("Fix simulation tracker querying
> occupancy with VAV IDs") then landed, and that fix changed simulation
> trajectories: querying occupancy with VAV ids "creates fictitious
> populations and consumes random draws, changing subsequent simulation
> trajectories". The trajectories moved; the table did not.
>
> **So the shipped table cannot be used as a reference for current code.**
> Every number this repository reports against a baseline is regenerated
> locally with `hvaclab.sweep`.

Two diagnosable pathologies in the strongest baseline, both worth more than
that oracle gap:

- **`fairbanks_1`**: `pid_water` scores 0.212 against `fixed_setpoint`'s 0.147.
  Its water reset and proportional reheat burn gas all night for comfort that
  is not being scored.
- **`singapore_4`**: `pid_water` scores 0.109 against `fixed_setpoint`'s 0.024,
  worse on all three terms and **9× the carbon**. In a climate that never needs
  heat, the worst-zone air-handler loop overcools some zones and the per-zone
  reheat valves then pay gas to warm them back up.

The score is a mean over 30 equally weighted scenarios, and the four worst
contribute ~38% of `pid_water`'s total. The tail is the target, not the mean.

## License / attribution

Apache-2.0, see `LICENSE`. `smart_buildings/` is the official starter kit and
builds on Google's open-source Smart Buildings Simulator; see `NOTICE`.
Competition materials belong to the organizers; this repository is an
independent entry.
