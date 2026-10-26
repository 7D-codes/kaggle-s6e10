# Score escalation evidence review

Owner: Codex / GPT-6. Branch: `main`, documentation only. Scope: this plan, its strategy subsection and one linked experiment-ledger entry. No runtime files, model fits, job launches or submissions. The owner's newly supplied AGENTS instructions pause new training; earlier resumption entries remain historical evidence.

## Objective and approach

Respond to the request for a substantially stronger result by checking the existing Kumo job and recent public notebook sources. Separate verified offline contribution from public leaderboard scores. Identify genuinely new evidence before proposing another experiment; do not change the live run's frozen control, thresholds or budget.

## Progress

Repository status clean; fetched and fast-forward pull reported already up to date at `2bf84ca255ac6457b2da7ebf80bc465b315e968d`. Read current rules, strategy, experiment ledger and onboarding. Existing private Kumo kernel 141062725/version 1 remained RUNNING with an empty failure message at 2026-10-26T23:40:35 UTC. No fold-completion or quality result is established by that status.

Current verified CatBoost meta contribution remains +0.00010496546923383286 development ROC AUC, five positive folds. Its fixed candidate remains unsubmitted. That number cannot be added to the reported Summit public score.

## Public-source findings

Four current public notebook snapshots were acquired read-only via Kaggle API and inspected without executing their code or loading their models. Numeric versions are unknown for these snapshots; source SHA256 identities below were independently rechecked. Web rendering failed, so the inspected evidence is the API-downloaded notebook source, not a rendered leaderboard verification. Copied notebooks stay in ignored research storage.

| Primary source | Source SHA256 | Finding and limit |
|---|---|---|
| [Meta Stack](https://www.kaggle.com/code/goodpjw2008/s6e10-meta-stack-on-public-oof-lb-0-96213), kernel137644080 | `9738a0413833d367d557b0282fb40c7eee82e152569c0b7a72d97c8748214d3a` | Author labels the reported0.96213 export as selected through public feedback, with no matching CV claim. Current comparison of leading files emphasizes near-duplicate predictions. It does not recover the exact Summit OOF or supply a newly qualified independent model. |
| [Twin Stacks](https://www.kaggle.com/code/goodpjw2008/s6e10-twin-stacks-family-reg-lb-0-96187), kernel137003482 | `a38056b29703004bdcd15b24618ae28ebbb54b1c1064d5c15597babb70205a3e` | Author reports a much stronger standalone four-estimator TabPFN but essentially no incremental contribution to their large pool. Pseudo-label CV gains failed public validation. These author-reported results concern a different ensemble and are not our replications. Family-specific regularization gains are small, not evidence of a breakthrough. |
| [Before you pick your final two](https://www.kaggle.com/code/georgymamarin/s6e10-before-you-pick-your-final-two), kernel137722750 | `632335a7d242284aac8c99a0e7a0b65f65021adf0fd7a1224cd70626faecd5c3` | Two rating-based specialists and three simple rating aggregates lost their respective comparisons. The source explicitly discloses reused labels and nonnested upstream predictions. Avoid treating those same generic feature ideas as untested opportunities; this does not reject all group-specific models. |
| [GBDT+TFM family study](https://www.kaggle.com/code/yangkuangou/s6e10-gbdt-tfm-cv-96202-lb-0-96169), kernel140997889 | `5f4c4dd2015af7bbcb91afd22a6450551a866d3dd1d86b40cb29e0ce348f1f7c` | Author reports38members, metaCV0.962021 and public0.96169 for submission57669665. Source acknowledges inherited outer-fold stopping, resplit base OOF and recipe-selection optimism. It is useful evidence of a CV/public gap, not a cleaner control or a new proven improvement over Summit. |

Read-only TFM dataset inventory contains the same22NPYfiles/11pairs already known locally. Public RealMLP/XGBoost were previously excluded from our strict pool for stopping provenance; an inventory count of11 is not two newly discovered eligible models. No additional predictions were imported or fitted.

## Decision and next evidence

No newly qualified complementary member emerged. Do not duplicate a large public-pool stack, sweep blend weights, repeat rejected neural/teacher arms or make an inference-batch change inside the live Kumo experiment. Higher leaderboard positions alone do not identify transferable new signal.

The existing job tests a specific representation: matched raw versus eight engineered Kumo features, with the qualified Cat-containing control held fixed. Its next useful evidence is a complete pair's measured feasibility, then complete five-fold/native/coverage results. A resource failure closes this configuration under its cap; it is not permission to extend the cap or silently restart. A complete valid run would be eligible for the already reviewed14vs13 and14vs12 comparisons only after the owner resumes evaluation under the newly supplied pause instructions. Those comparisons retain their fixed gains/fold/conditional-interval gates; no leaderboard gain is promised.

If that experiment fails its quality gates, the next council should choose one distinct representation or architecture with a concrete information argument against the twelve-member control. More seeds/estimators or larger context are not automatically the next experiment. The earlier TabM proposal remains unverified and unlaunched; no successor configuration was selected in this review.

## Verification and handoff

All four notebook file hashes matched the stored manifest; source narratives and meta-validation limitations were inspected. Documentation whitespace, local links and staged-file scope checked before publication; no runtime code changed, so no runtime tests or model fits were run. No new job, duplicate launch, submission or model download occurred. No new public improvement is established. Final read-only status at 2026-10-26T23:42:40 UTC remained RUNNING with an empty failure message. Updated the strategy's project-state banner to the latest pause so historical authorization cannot trigger a teammate launch. Continue monitoring the exact existing version and preserve terminal artifacts. Ownership is released after publication; training/evaluation resumption remains the owner's decision.
