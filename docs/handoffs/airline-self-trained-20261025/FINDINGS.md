# Findings

## Outcome

Our best submitted full-data ensemble scored **0.96119** publicly. Earlier own submissions scored 0.96112 (original frozen ensemble) and 0.96105 (single). The copied historical 0.96213 entry was excluded and is not a campaign result. The exact end-to-end solution behind that copied ensemble was not reproduced.

The best submitted recipe ranks each of five primary model outputs over the 299844 test rows, divides by the test row count, then sums:

| Primary recipe | Weight |
|---|---:|
| cat_twins_tabpfnteacher_v1 | 0.1 |
| reference_catboost | 0.1 |
| nested_stack_reference233_services13_lgbm021_cat_twins_v3 | 0.3 |
| nested_stack_reference233_services13_lgbmteacher_catteacher_v5 | 0.4 |
| realmlp_lean_sigma2.33_twins_epochs6_lr025 | 0.1 |

The new numeric CatBoost component used 1500 iterations, depth 7, learning rate 0.05, border count 128 and L2 leaf regularization 3. Its 70 fold-local features include cross-fitted target encodings, original numeric/rating columns, ordinal categories, a training-only cluster and an arrival-missing indicator. Full refitting used 699635 labeled competition rows.

Replacing the original ensemble's fixed 0.1 LightGBM leg with this CatBoost leg improved paired DEV AUC by 0.0000325395 and 0.0000302192. All ten fold differences were positive and conditional paired intervals were positive. The actual upload improved public AUC from 0.96112 to 0.96119.

## Completed reference trials

These are local DEV OOF AUCs, not public leaderboard results. Each completed model also produced all 299844 test predictions and passed exact saved-state replay.

| Trial | OOF assignment 42 | OOF assignment 2026 | Decision |
|---|---:|---:|---|
| CPU LightGBM on reference features | 0.96050708 | 0.96050481 | Fixed blend screen failed |
| TabPFN V1: 100k context, 1 estimator, batch 64 | 0.95958886 | 0.95969700 | Fixed blend screen failed |
| TabPFN V3: 150k context, 1 estimator, batch 128 | 0.95981127 | 0.95994684 | Fixed blend screen failed |
| TabPFN V4: 65536 context, 2 estimators, batch 256 | 0.95946117 | 0.95963508 | Fixed blend screen failed |
| Numeric CatBoost V5 | 0.96058594 | 0.96054775 | Fixed rank replacement qualified; uploaded ensemble 0.96119 |
| Native-category CatBoost V6 | 0.96053100 | 0.96054011 | Both fixed tests versus uploaded V5 failed |
| XGBoost V7 | 0.95979960 | 0.95989781 | Both fixed tests versus uploaded V5 failed |

A separately frozen five-recipe reference consensus lost on both assignments in both probability and rank forms. Larger TabPFN contexts were rejected on measured GPU memory or guarded runtime before quality evaluation. These bounded local adaptations were not identical reproductions of the public reference configuration. Increasing query batch changed probabilities and was not an equivalent inference substitution.

## Validation limits

Both five-fold assignments reuse the same DEV rows; they are not independent datasets. Conditional paired intervals do not correct adaptive experiment selection. Twenty-one subgroups had no interval-excluded losses for the uploaded V5 candidate, but small cohorts (including n=148 and n=249) remain uncertain. Some subgroup point estimates were negative. The original final holdout was revealed once and was not rescored for the reference trials.

All reference workers, full refits and fixed reviews finished. No failed or already scored recipe was reset or rerun for this handoff. No public score was guaranteed. The authorized training campaign ended with the requested public target unmet; packaging these existing files does not restart training.
