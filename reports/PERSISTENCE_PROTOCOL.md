# Cooling persistence on Kaggle Version 7

Prospective 2026-11-01. Best reported public score: V7/valve8_gentle, 0.01922.
Retain its global valve h**0.75, PI, damper, AHU/water coordination and schedule.
Change only cooling hold duration: 144, 432, 576 five-minute steps (12, 36,
48 hours), against the unchanged 288-step (24-hour) control. New cooling
requests refresh the timer; heating cancels it, exactly as the existing code.
This changes persistence after the latest cooling request, not necessarily
total uninterrupted runtime. Unlike earlier night-only studies, it changes
the timer throughout the episode on the now organizer-confirmed valve curve.

Hypothesis: a shorter hold may save unnecessary ventilation while a longer
hold may preserve useful continuous delivery. Neither direction is assumed
beneficial. No geometry, weather/building labels, or test split is used.

One fresh full V7 control must match cached reward within 1e-10. All three
candidates receive all 20 dev scenarios, then eligible candidates receive all
30 training scenarios in dev-rank order, continuing after a failure. Require
both full-cohort means lower than V7 AND V4 by more than 1e-10 before packaging
and standalone parity checks. Maximum 151 full episodes before parity checks.
Report individual regressions and components; repeated dev tuning is exploratory
and only Kaggle can establish a score below 0.01922.

Runner: hvaclab.persistence_calibration. Run directory:
experiments/runs/persistence_calibration. Frozen source/reference hashes,
three workers maximum, 20-second spacing, launch thresholds 1.75 GiB available
RAM and 1 GiB disk. Independent supervisor launched with:

```powershell
powershell -NoProfile -File reports/scripts/start_study_supervisor.ps1 -Study persistence_calibration
```

Preserve all previous artifacts/results; append paper section and table.
Supervisor resumes from valid records if scheduler disappears; no reboot startup
or sleep prevention is installed.
