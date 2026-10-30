# V6 organizer feedback

Source: user's Kaggle screenshot in this conversation, recorded 2026-10-30.

| Version | Displayed public score |
|---|---:|
| V4 | 0.01984 |
| V5 | 0.01996 |
| V6 | 0.03833 |

Lower is better. V4 remains best among these submissions; V6 is approximately
93.2% worse than V4. Local full-cohort improvements have not established an
organizer improvement. Retain prior local measurements and submission files.

V6 and V3 have the same displayed score to five decimal places. This alone
does not identify the cause or establish that the wrong notebook was uploaded.
The local V6 notebook hashes match its manifest/validation, embeds
`geo6_threequarter`, and its actual inference wrapper matches 500 synthetic
action pairs against the registered controller. See `V6_LOCAL_NOTEBOOK_AUDIT.json`.
This checks the local artifact, not the uploaded Kaggle Version 6, its data,
metadata, logs or evaluation execution; those remain unverified.

The frozen V7 experiment remains exploratory. A local pass will not justify
a claim that geometry-based controllers outperform V4 on organizer evaluation.
