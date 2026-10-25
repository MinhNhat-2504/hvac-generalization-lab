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
