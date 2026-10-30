# V8 candidate study: reheat-valve curve on the V4 controller

Prospective protocol, 2026-10-31. V4/e_sync remains the best known Kaggle
submission: public 0.01984 (V5 0.01996; V6 0.03833). Lower is better.
The V7 split-geometry study completed 91 episodes but failed training; no
notebook was packaged. Preserve all prior results and artifacts.

## Hypothesis and fixed variants

Return to V4 without geometry scaling. V4's signed zone PI demand u commands
damper abs(u) and valve h=max(0,-u). The simulator's coil heat depends on
water capacity, air capacity and discharge-temperature caps; it is not simply
proportional to the product of damper and valve. A different valve response
may improve the comfort/energy trade while preserving V4 air delivery.

Change only the valve to h**p during active zone PI:

| Key | p | Response |
|---|---:|---|
| valve8_ctrl | 1 | Exact V4 reproduction |
| valve8_sqrt | 0.5 | More valve opening at intermediate heating demand |
| valve8_gentle | 0.75 | Gentler increase in opening |
| valve8_soft | 1.5 | Less opening at intermediate heating demand |

Every curve retains h=0 and h=1, so zero/full valve authority is preserved.
PI gains, raw integral limits, conditional integration and signed u remain
unchanged, as do damper commands, AHU control, pump/water coordination, bands,
one-hour prestart and 288-step cooling hold. Outside the PI window, inherit
the V4 law. No room map, building name or weather label determines the rule.
Retaining the PI state law does not guarantee unchanged closed-loop dynamics:
the altered valve characteristic is the mechanism under test.

## Evaluation and selection

1. Require a new full 8,640-step identity-curve control to reproduce the cached
   V4 reference within 1e-10 mean reward.
2. Evaluate all three fixed candidates on all 20 dev scenarios (60 episodes).
3. Rank full-dev means lower than V4 by more than 1e-10. Evaluate all 30 training
   scenarios for eligible candidates in rank order, continuing after a failure.
4. Package only a candidate with both complete cohort means below V4, then
   verify standalone actions and save hashes, comparisons and description.
   Otherwise retain V4 and record the negative outcome.

Maximum 151 full episodes before standalone verification. Equal scenario
weights, complete matched episodes; report component costs, regression counts
and largest increases. Do not compare partial cohorts or short horizons as
performance evidence. No arbitrary individual-scenario veto is added.

Dev has been repeatedly tuned and training is published data. Neither is
untouched validation. The separate local test split stays unused. A local
pass permits an experimental submission, not a claim to have beaten 0.01984.
Only a new organizer result establishes that outcome.

## Artifact and runtime checks

The previous V6 local notebook audit checked artifact hashes and 500 actual
inference-wrapper action pairs. Uploaded Kaggle Version 6, metadata and logs
remain unverified; the displayed score matching V3 does not identify a cause.
Current tracked simulator files have no working-tree changes, which does not
establish that they match the organizer runtime version.

New valve-curve notebooks print `HVAC_SUBMISSION variant=... source_lf_sha256=...`
at startup. The hash covers the assembled source encoded as UTF-8 with LF;
it differs in definition from a Windows CRLF file hash. This gives a concrete
marker to compare in Kaggle execution logs before interpreting a future score.
Tests execute the full generated notebook wrapper with a stub inference server
and compare actions/marker to the registered policy. These tests do not claim
to reproduce the actual Kaggle service.

## Execution and paper

Run `python -m hvaclab.reheat_curve_calibration` from the repository root.
Outputs: `experiments/runs/reheat_curve_calibration`; at most three independent
workers, 20-second spacing, launch thresholds 1.75 GiB RAM and 1 GiB free disk.
Freeze source/config/reference hashes and dependency versions. Resume only
matching valid per-episode results; use the duplicate-run lock and atomic JSON.
Build the paper in a separate directory and replace the public PDF only after
successful compilation. Add this protocol and a separate results table while
retaining every previous measurement and submission artifact.
