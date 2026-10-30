# e_sync broad coverage

Run status: **complete**. Updated: 2026-10-27T07:32:44+0700.

The submitted Version 4 notebook is unchanged. These are local full-episode measurements, not a Kaggle result. The test split is not used.

| Split | Completed | Expected |
|---|---:|---:|
| training | 30 | 30 |
| dev | 20 | 20 |

Aggregates in the run directory may be partial. A full-set comparison appears only after every expected scenario in that split finishes.

## training: 30 matched scenarios

Score: h_288 **0.02774466** → e_sync **0.01445394** (-47.90%; lower is better). Improved 24, regressed 5.

| Policy | Comfort | Energy | Carbon |
|---|---:|---:|---:|
| h_288 | 0.01576395 | 0.00705920 | 0.00492151 |
| e_sync | 0.00201534 | 0.00702192 | 0.00541668 |

Largest increases (negative values are improvements):

| Scenario | h_288 | e_sync | Difference |
|---|---:|---:|---:|
| building_train_2/phoenix_6 | 0.01887899 | 0.01907323 | +0.00019424 |
| building_train_1/phoenix_5 | 0.03123583 | 0.03142563 | +0.00018980 |
| building_train_8/singapore_7 | 0.01261441 | 0.01271383 | +0.00009941 |
| building_train_5/singapore_4 | 0.01232418 | 0.01236581 | +0.00004163 |
| building_train_4/singapore_3 | 0.01573521 | 0.01574059 | +0.00000538 |
| building_train_1/singapore_8 | 0.01469297 | 0.01469297 | +0.00000000 |
| building_train_2/singapore_1 | 0.01177986 | 0.01177421 | -0.00000564 |
| building_train_4/atlanta_5 | 0.01519634 | 0.01517435 | -0.00002200 |
| building_train_5/atlanta_6 | 0.00997598 | 0.00976467 | -0.00021131 |
| building_train_7/phoenix_3 | 0.01655300 | 0.01620892 | -0.00034408 |

## dev: 20 matched scenarios

Score: h_288 **0.03449141** → e_sync **0.01668440** (-51.63%; lower is better). Improved 17, regressed 3.

| Policy | Comfort | Energy | Carbon |
|---|---:|---:|---:|
| h_288 | 0.02207958 | 0.00646678 | 0.00594505 |
| e_sync | 0.00341916 | 0.00659122 | 0.00667402 |

Largest increases (negative values are improvements):

| Scenario | h_288 | e_sync | Difference |
|---|---:|---:|---:|
| building_9001/singapore_2 | 0.01177561 | 0.01190436 | +0.00012875 |
| building_9002/atlanta_4 | 0.01783121 | 0.01791738 | +0.00008617 |
| building_9000/atlanta_4 | 0.01551861 | 0.01552070 | +0.00000209 |
| building_9002/singapore_2 | 0.01997681 | 0.01985976 | -0.00011704 |
| building_9001/atlanta_4 | 0.01034346 | 0.01020130 | -0.00014216 |
| building_9000/singapore_2 | 0.01756019 | 0.01735312 | -0.00020707 |
| building_9003/atlanta_4 | 0.01125844 | 0.01095803 | -0.00030040 |
| building_9003/singapore_2 | 0.01135646 | 0.01104012 | -0.00031634 |
| building_9002/phoenix_2 | 0.01663803 | 0.01502904 | -0.00160899 |
| building_9001/phoenix_2 | 0.00943731 | 0.00690758 | -0.00252973 |

## Reproducibility

Resume with `python -m hvaclab.coverage --workers 2` from the repository root only after checking that the existing coverage process has stopped. Do not launch two schedulers for the same output directory.

Each job has an isolated interpreter, log, descriptor and atomically saved result. The provenance file freezes policy/simulator/runner sources, scenario gin files, and the two explicitly reused caches. Failed or short runs cannot complete coverage.

Source/config/cache hashes protect against code drift; external weather and geometry data files are not hashed by this runner and must remain unchanged during the sweep.
