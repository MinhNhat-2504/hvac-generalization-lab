# Earlier PI preconditioning

Status: complete; stage: hypothesis_not_supported.

All scores use complete 8,640-step episodes against matched e_sync references.
The diagnostic episode is explicitly reused as the control, not counted twice as independent evidence.

| Cohort | Variant | n | e_sync | Candidate | Change |
|---|---|---:|---:|---:|---:|

Decision:

```json
{
  "first_two_hours_comfort_share": 0.12004407059406479,
  "diagnostic_supports_morning": false,
  "stop_reason": "first-two-hour comfort share below 25%; do not run the proposed prestart sweep"
}
```

Earlier paper sections and the e_sync public score remain unchanged. Screens are subsets of full cohorts, not additional independent evidence.