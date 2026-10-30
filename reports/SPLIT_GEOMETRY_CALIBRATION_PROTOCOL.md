# V7: separate proportional and integral geometry sensitivity

Prospective study, 2026-10-30. V6 is submitted and its Kaggle score is pending.
V4 remains the best known organizer score, 0.01984; V5 scored 0.01996.
V6 passed local checks at 0.0140783514 training and 0.0160495740 dev, reductions
of 0.556% and 0.264% versus V5, but only 8 of 20 dev scenarios improved.
All measurements remain in the earlier reports and paper.

## Hypothesis and fixed candidates

V5 and V6 multiply both proportional and integral gains by the same room-area
factor. This coupling need not be appropriate: P acts on current error and I
accumulates error across time. V6 reduces dev weighted comfort penalty versus
V5 (0.0026861601 versus 0.0027456800), but carbon rises slightly (0.0067357395
versus 0.0067092221). Test separate sensitivities rather than another coupled
exponent. These observations motivate a hypothesis, not a causal proof.

Let sp = clip((A/2484)^p, 0.5, 2), si = clip((A/2484)^i, 0.5, 2).
Use Kp = 0.1sp, Ki = 0.005si, raw integral limit = 200/si. Maximum integral
contribution remains one. Area reference and mapping are unchanged from V5/V6.

| Variant | p | i | Interpretation |
|---|---:|---:|---|
| geo7_ctrl | 0.75 | 0.75 | V6 reproduction |
| geo7_soft_i | 0.75 | 0.5 | Weaker integral geometry sensitivity |
| geo7_strong_i | 0.75 | 1.0 | Stronger integral geometry sensitivity |
| geo7_soft_p | 0.5 | 0.75 | Weaker proportional geometry sensitivity |

The subclass reuses the existing PI integration/actuator law by compensating
the integral gain and state-limit ratio before the unchanged geometry layer.
It restores attributes after every command. Equal exponents reproduce V6;
missing/inconsistent geometry uses the existing fallback. No simulator/reward,
building name or weather label determines runtime actions. Existing mapping
limitations remain documented in `GEOMETRY_CALIBRATION_PROTOCOL.md`.

AHU temperatures and pressure, pump/water coordination, comfort bands,
one-hour prestart, 288-step cooling hold, PI activation and conditional
integration remain unchanged. This study changes no plant control settings.

## Evaluation and packaging

1. A fresh 8,640-step control must reproduce V6 within 1e-10 mean reward.
2. Test all three fixed candidates on all 20 dev scenarios (60 episodes).
3. Rank candidates whose full-dev mean improves on V6, V5 and V4 by more than
   1e-10. Evaluate all 30 training scenarios in that order, continuing after
   a training failure.
4. Package the first candidate whose full training mean also beats all three
   references. Check standalone action parity and save hashes, comparisons and
   submission description. Otherwise retain existing artifacts and record failure.

Maximum 151 full episodes plus standalone checks if selected. Compare complete
matched cohorts, equal scenario weights. Report individual regressions without
a hard worst-case veto. Dev is repeatedly tuned model-selection data and
training is a published consistency check. Neither is untouched confirmation;
the separate local test split stays unused. A local pass cannot establish a
Kaggle improvement, as V5 organizer feedback already demonstrates.

Freeze code, configs, geometry maps, all three score references and dependency
versions. Use `experiments/runs/split_geometry_calibration`, isolated workers,
a duplicate-run lock, atomic saved records and matching-source resume. At most
three workers, 20-second launch spacing, 1.75 GiB RAM and 1 GiB disk available
before each launch. Paper builds use a separate output directory and replace
the published PDF only after successful compilation. Add a new section/table
without removing any prior data or changing V4/V5/V6 submission artifacts.
