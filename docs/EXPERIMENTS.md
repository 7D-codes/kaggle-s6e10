# Shared Experiment Ledger

This is the public-safe handoff, not a replacement for private full predictions/logs. All OOF below is ROC AUC on development-exposed rows, not untouched holdout. Experiment status does not grant permission to resume training.

## Retained submission

- Public ROC AUC: **0.96111**, submission **57568517**, COMPLETE.
- Fixed weights: 50% baseline RealMLP, 25% full-route LightGBM, 25% full-route XGBoost.
- Submitted prediction SHA256: `5ac86fa1164abc08a0483c97b9732caa0b43b2284a4e54bd17590d162f241097`.
- Rows: 299844; split seed: 2026. No label-fitted weights or averaging across split seeds.
- Local evidence: `reports/batch6-candidate-receipt.json`, `reports/submission-57568517.json`, `results/batch6-confirm-neural-blend/`. These paths are ignored and not available in a fresh clone.
- Submission authorization was consumed. No further submission is authorized.

## Batch 6: accepted development improvement

| Split | Fixed candidate OOF | Matched incumbent OOF | Gain | Positive folds |
|---|---:|---:|---:|---:|
| 42 | 0.9614620347107486 | 0.9613172788969397 | 0.00014475581380890556 | 5/5 |
| 2026 | 0.9614641415242471 | 0.9613436210433547 | 0.00012052048089239431 | 5/5 |

Five-fold stratified shuffle uses the stated split seed. The seed-2026 control retained the same neural member and matched teacher-LightGBM/rich-XGBoost components. Repeating on reused rows supports robustness, not pristine generalization. The recipe uses fold-local route/frequency features, digit features, cross-fitted TE, source-only priors and an overlap-filtered source-only teacher. Preserve original teacher/vocabulary in native inference bundles.

Remote sources, each completed private version 1 (access is not implied by a link):
- [Seed-42 route benchmark](https://www.kaggle.com/code/wasidyms/s6e10-batch6-route-recipe).
- [Seed-2026 confirmation](https://www.kaggle.com/code/wasidyms/s6e10-batch6-route-confirmation), kernel140271383, source SHA256 `5d715c2bb38ffb2cd8e8477f4f31c6bceafaf4d03005444d6333edaa07345528`.
- [Checkpoint replay audit](https://www.kaggle.com/code/wasidyms/s6e10-batch6-checkpoint-replay-audit).

Confirmation runtime pins: NumPy2.1.3; route pandas3.1.0/sklearn1.9.1; LightGBM4.7.0; route XGBoost3.2.0. Matched historical XGBoost control used XGBoost3.4.1/pandas2.3.3/sklearn1.6.1. Model versions are arm-specific, not global project pins.

### Inference caveat, do not erase

New fitted states freeze float32 rarity outputs to prevent runtime-dependent logarithm differences crossing tree thresholds. Historical states remain unchanged. Confirmation CUDA raw-row spot replay: both tree arms exact on64rows. Strict CPU XGBoost raw-row replay at `rtol=0, atol=1e-7` failed1/128rows, maximum `1.043081283569336e-7`. CPU equivalence is not accepted. Exact LightGBM replay passed128rows. These are spot checks, not full-test inference proof. Loading Arrow-backed joblib states may require PyArrow.

## Batch 7: completed, rejected, no confirmation/submission

Same seed42 folds and fixed50/25/25blend, with fresh3epoch control. Gate: gain≥0.00002 and≥4/5positive paired folds against both fresh and historical controls.

| Neural arm | Neural OOF | Fixed-blend OOF | Gain vs fresh control | Positive folds |
|---|---:|---:|---:|---:|
| Fresh3epoch control | 0.9611684784104386 | 0.9614593306389816 | 0 | control |
| Six epochs | 0.9607101760148082 | 0.9613980437561279 | -0.00006128688285367279 | 0/5 |
| Applicability summaries | 0.9611726718948578 | 0.9614675645954114 | 0.000008233956429792855 | 3/5 |

Decision: retain Batch6, no seed2026 confirmation. Do not recycle these arms without materially new evidence. Six epochs also changes the epoch-normalized learning-rate schedule, so this does not prove a universal effect of training longer.

[Private completed kernel](https://www.kaggle.com/code/wasidyms/s6e10-batch7-neural-improvements), version1. Full neural checkpoints intentionally remain remote, not in Git. CUDA64row raw-row replays passed at original `rtol=1e-6, atol=1e-7`, max error `1.1102230246251565e-16`. PyTabKit1.7.3, torch2.11.0+cu128, NumPy2.1.3, pandas2.3.3, sklearn1.6.1, cloudpickle3.1.2. Full local evidence: `reports/batch7-results.md`, contracts/comparison/readback, and `results/batch7-neural-improvements/`.

## Other completed approaches

- Original-source target priors and source-only teacher improved selected tree controls.
- Raw tree averaging did not reliably improve the stronger baseline.
- Teacher-enhanced neural and source-only auxiliary-rating final blends did not beat the retained controls; rejected.
- None of these is an unexplored default task for a new agent.

## Diagnostic reset: research only

- Parent independently verified seed2026 AUC0.9614641415242471 and route-tree prediction correlation0.9997051652128117.
- Saved-OOF rank-error identities and nine broad partitions were audited. Hard cases include satisfied personal travelers and dissatisfied high-service-rating passengers. These are exploratory groups, not evidence that subgroup-specific modeling improves generalization.
- Inspected0.962+public notebooks mainly recombine inherited predictions and/or select against public scores. Their claimed scores are not reproduced training results.
- Distinct possible lead: [independently trained full-context TabPFN-3.5](https://www.kaggle.com/code/hemingweb/s6e10-exp01-tabpfn35-full-context). Inspected source snapshot SHA256 `6f683e494124f1169a413bb38646b046d1e3362602d9893e3fb21ad95a013c82`; numeric kernel version unknown. Its fold-separated training exists, but clean OOF gain against our incumbent is unverified. It includes a large11.5hour scheduler and a custom KV patch whose equivalence is not independently established. Do not execute it wholesale.
- Training remains PAUSED. No diagnostic work imported public predictions, trained a model or submitted.

## Codex research update — 2026-10-24, no experiments launched

See [breakthrough research and proposed gates](plans/codex-breakthrough-research.md) for primary sources, source hashes, validation risks and a ranked queue. Independent TabPFN-3.5 contribution is the first hypothesis; TabM is a lower-evidence second architecture. No gain is established. Current public 0.96209 source explicitly uses leaderboard-driven selection, so it is not proof of a clean modeling improvement.

Hemingweb's previously recorded source hash was matched through the public API and its current numeric version identified as **2**. Inspection also found train/test-wide category mapping and potentially uneven or partial fold/seed aggregation; any adaptation must use fold-local mappings and complete balanced outputs. The custom memory patch remains unverified. No historical result or replay tolerance changes.

All proposal gates are conditional on explicit training resumption and a finalized experiment contract. No training, GPU smoke, remote job, submission or use of the supplied credential occurred. Incumbent 0.96111 remains retained.

## Contract template for later authorized experiments

### Codex resumption — 2026-10-26

Owner explicitly resumed training/evaluation after the pause discussion. [Frozen CatBoost comparison and evidence](plans/codex-resumed-catboost-comparison.md): eleven-member control versus identical control plus complete audited CatBoost; twenty same-original-fold CPU meta fits, unchanged seed/configuration/gates. Primary developmentROC AUC **0.9616869424637373→0.9617919079329712**, gain **0.00010496546923383286**,5/5positive folds; conditional95%gain interval[0.00007762420,0.00014243317]. Passed frozen gate in47.9543seconds. Independent full-coverage/coefficient/AUC/fixed-candidate audit passed. Fixed UNSCORED Summit correction SHA256 `93282e7daa51ffa02f411fa186324edbb83143df49790933a6415087721cbd6b`; no exact-Summit/public improvement established. No CatBoost retraining, GPU launch or competition submission in this comparison.

Record before launching: agent/owner, hypothesis and evidence, control, exact fold seed, parameters/feature changes, fixed blend weights, versions/checkpoint/source hashes, execution location, resource/time cap, promotion and stop criteria, job/kernel version and monitor owner. After completion append actual device/smoke status, complete OOF verification, paired gains, replay results/caveats, decision, artifact location/access and submissions used. Preserve rejected results; never silently replace a historical record.
