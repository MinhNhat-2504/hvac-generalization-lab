# Geometry-scaled PI: prospective protocol (2026-10-29)

Goal: improve the complete local training and dev mean scores relative to the
unchanged V4/e_sync controller (public score 0.01984). A local improvement is
evidence for submission, not a prediction or guarantee of a public score.

The previous scalar PI study finished 251 complete episodes. Its three
dev-improving candidates all regressed on full training: firm proportional
+1.669%, fast integral +3.964%, and firm/slow +0.991%. Those results remain in
`PI_CALIBRATION_REVIEW.md` and the paper. No replacement notebook was selected.

Hypothesis: equal gains across unequal zone sizes can cause different thermal
responses. Current VAV maximum airflow is uniform while zone thermal mass
depends on geometry. Scale zone gains using participant-visible room area;
keep every other V4 setting, including plant coordination and cooling hold.
This is a hypothesis, not a proven causal explanation of the preceding results.

## Fixed rules

For zone area A in control volumes, s = clip((A/2484)^p, 0.5, 2).
The reference 2484 is the median of 323 zones in the six training buildings.
Dev geometry verifies mapping only; it does not determine this reference.

| Arm | p | Kp | Ki | Raw integral limit |
|---|---:|---|---|---|
| geo_ctrl | disabled | 0.1 | 0.005 | 200 |
| geo_p_sqrt | 0.5 | 0.1 s | 0.005 | 200 |
| geo_pi_sqrt | 0.5 | 0.1 s | 0.005 s | 200/s |
| geo_p_linear | 1 | 0.1 s | 0.005 | 200 |

Maximum integral contribution remains 1 for each zone. Conditional integration,
PI activation, comfort bands, one-hour prestart, and 288-step (24-hour) hold
remain unchanged. No building/weather identifier chooses the gains.

The runtime uses NumPy and Python to count four-connected interior components
(map value 0) in raster-root order. An offline SDK/OpenCV check verifies exact
areas and VAV-to-room numbering on all ten current training/dev buildings;
see `experiments/plans/geometry_reference.json`. Component scans take under
0.1 second per current map. Runtime uses only participant metadata. Missing maps
or a mismatch between component count/expected VAV IDs and observed VAV IDs
fall back to V4. Label-order compatibility on other SDK versions is not proven;
a same-count permutation would not be detected by the runtime fallback.

## Measurement and selection

1. Run one fresh 8,640-step disabled-geometry control and require its mean reward
   to match the existing V4 reference within 1e-10.
2. Run all three rules on all 20 dev scenarios (60 full episodes).
3. Rank rules with lower full-dev means. Evaluate all 30 training scenarios for
   each eligible rule in order, continuing after any training failure.
4. Package the first rule with both full means lower by more than 1e-10. Verify
   standalone action parity, record notebook/source hashes, and write the
   submission description. Otherwise retain V4 and record the negative result.

At most 151 new full episodes, plus standalone verification if selected. Scores
are equal-weight means over complete matched scenarios. Individual regression
counts and largest increases remain visible; no hard worst-case cutoff is used.
Dev has been repeatedly used for model selection; neither training nor dev is
an untouched confirmation set. The separate test split remains unused.

The scheduler freezes code/config/reference hashes and dependency versions,
uses a duplicate-run lock, checkpoints each rollout, and starts at most three
workers with a 1.75 GiB available-memory launch threshold and 20-second spacing.
The paper receives a separate results table; all preceding measurements stay.
