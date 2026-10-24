# Codex: research toward public ROC AUC above 0.96200

## Scope and ownership

Owner: Codex / GPT-6. Branch: `main`. Research only; training and submissions remain paused. Changes reach this plan, Codex's strategy entry and a research-only experiment-ledger link. No runtime callers, fixtures, configurations or exports change. Preserve historical results, private artifacts and other agents' entries.

## Strategy

1. Read repository coordination, experiment history, project and onboarding documents.
2. Check current competition pages and primary model/notebook sources; separate measured public scores, author claims, inspected methods and our own verified evidence.
3. Rank a small set of complementary model hypotheses and specify later validation/resource gates; do not recycle rejected ideas without new evidence.
4. Publish findings, limitations and handoff. No model fitting, GPU smoke, remote job or submission.

## Initial evidence and progress

- Initial worktree clean. Fetched and fast-forward pulled `main`; read `origin/main` coordination documents.
- Retained public score 0.96111; development OOF 0.9614641415242471. Target is above 0.96200, requiring more than 0.00089 public improvement.
- Route-tree prediction correlation 0.9997051652128117 is historical ledger evidence, not recomputed here.
- Longer neural training, applicability summaries and teacher-enhanced neural features already failed promotion gates.
- Local ignored reports/results are absent; existing numerical evidence currently comes from the shared ledger.

## Research conclusion — 2026-10-24

The best next hypothesis is an independently reproduced TabPFN-3.5 member, judged by its incremental gain over our retained blend. TabM is a second, lower-evidence architecture candidate. Only after obtaining complementary predictions should we consider a learned combiner. None is a demonstrated route to public ROC AUC above 0.96200 yet.

The public-score gap is greater than 0.00089 (0.00090 to displayed 0.96201). That is much larger than Batch 6's roughly 0.00012–0.00014 development improvement. Different evaluation populations prevent direct extrapolation, but the scale argues against expecting one tiny feature tweak to finish the task. High tree prediction correlation motivates testing diversity; it does not justify dropping a tree without a paired ablation.

## Competition and evidence boundaries

- [Kaggle's competition overview](https://www.kaggle.com/competitions/playground-series-s6e10/overview/about-the-tabular-playground-series) specifies binary satisfaction probabilities and ROC AUC, with the final deadline October 31, 2026 at 23:59 UTC. The task is ranking passengers, not maximizing thresholded accuracy.
- Public source reports 699,635 training rows; the ledger records 299,844 test rows. Our feature modules use 21 raw predictors: four categorical fields, age/distance/two delay columns, and 13 ratings. No local data files were available to independently recount rows.
- Incumbent is fixed 50% RealMLP / 25% route LightGBM / 25% route XGBoost. Seed-2026 OOF is 0.9614641415242471; public is 0.96111. Do not treat their difference as a universal conversion factor for future models.
- All historical OOF rows are development-exposed. A new seed or a newly reserved slice of those same rows is not retrospectively untouched data.
- Public notebook performance below is author/platform-reported, not independently reproduced. Source inspection verifies what code says/does, not that every claimed result came from those exact bytes.

## What current primary sources actually support

| Source | Observed evidence | Decision for this project |
|---|---|---|
| [Hemingweb full-context TabPFN](https://www.kaggle.com/code/hemingweb/s6e10-exp01-tabpfn35-full-context), API version 2 | Five-fold seed 42; all other-fold rows as context; raw features; `tabpfn==9.0.0`; custom KV-cache patch; 11.5-hour scheduler | Strongest directly relevant independent-model lead; adapt and verify, do not execute wholesale |
| [Goodpjw2008's series](https://www.kaggle.com/code/goodpjw2008/s6e10-twin-stacks-family-reg-lb-0-96187), API version 13 | Author reports TabPFN with four estimators at 0.961491 OOF and 0.96110 public, but no incremental gain in their 245-member stack; series' retained blend reports 0.96187 public | A strong single model can be redundant. Measure contribution to our three-member incumbent, not their much larger pool |
| [Rabari's 0.96209 notebook](https://www.kaggle.com/code/rabari9999/s6e10-lb-0-96209-rank-10), API version 55 | Source explicitly describes fitting moves to measured public scores; no OOF evidence behind those moves | Demonstrates a reported public number above the target, not an independently validated modeling breakthrough |
| [Deotte foundation-model stack](https://www.kaggle.com/code/cdeotte/stacking-tfms-with-public-oof), API version 1 | Consumes public OOF/test arrays and applies regularized logistic regression to logits; does not train its foundation-model members | Useful family inventory. Requires member-level provenance and validation audit before reuse; array shape is insufficient |
| [TabM official implementation](https://github.com/yandex-research/tabm) and [ICLR paper](https://arxiv.org/abs/2410.24210) | Parameter-sharing MLP ensemble; supports numerical feature embeddings and large datasets | Plausible second architecture; no verified S6E10 uplift found in this review |
| [Dariush's label-noise study](https://www.kaggle.com/code/dariushafshar/s6e10-the-gap-is-3-8-label-noise-not-features), API version 7 | Author fits a random-flip analogy and explicitly qualifies its uncertainty; reports concentrated rank errors involving disagreement with source-only predictions | A diagnostic hypothesis, not a proven Bayes ceiling or permission to delete/relabel difficult examples |

[Prior Labs' current model documentation](https://docs.priorlabs.ai/models) lists up to 1,000,000 training rows for TabPFN-3.5. That makes the roughly 560k-row training folds nominally eligible, not guaranteed to fit on a T4. Its [memory guidance](https://docs.priorlabs.ai/troubleshooting/OOM-errors) documents cached prediction, smaller query batches, memory-saving mode and context subsampling. Subsampling changes the statistical experiment; it must be named and recorded, not silently substituted for full context. Kaggle authentication does not by itself grant access to gated model weights.

The [S6E9 second-place write-up](https://www.kaggle.com/c/playground-series-s6e9/writeups/2nd-place-solution) provides adjacent-task support for full-context TabPFN and a warning: its leaderboard-fitted file did much worse privately than publicly. Its generator features and quantitative gains are not evidence that the same recipe works on airline satisfaction.

## Ranked research and experiment queue

### 0. Recover the exact comparator before spending compute

Obtain owner-authorized seed-42 and seed-2026 incumbent OOF/test artifacts from the ledger's private locations, using the owner's normal access. Check hashes, unique IDs, target alignment, exact fold membership, full coverage and probability validity with existing verification patterns. Preserve submission 57568517 and its prediction hash. No checkpoint deserialization is needed to compare CSV predictions.

Do not repeat the historical nine-partition audit as if new. Once a candidate exists, compare its paired ranking changes against the incumbent, including personal/business travel, source-teacher confidence and rating patterns already identified in the ledger. A subgroup defined using its observed label is an analysis group, never an inference routing rule. Overall paired AUC remains the selection criterion; correlation and subgroup gains are diagnostics.

### 1. TabPFN-3.5: highest-priority independent member

Hypothesis: in-context predictions capture relationships missed jointly by RealMLP and route trees. Start with 21 raw features, one estimator, one fixed model seed and identical seed-42 five-fold partitions. Preserve full other-fold labeled context where feasible. Evaluate `0.8 * incumbent + 0.2 * TabPFN` as the predeclared primary candidate; report standalone TabPFN as secondary evidence, not as a replacement decision. Those weights are a research proposal, not tuned or claimed optimal.

Specific corrections required by the inspected source:

- Learn category mappings inside each training fold; its current mapping uses concatenated train/test categories.
- Prefer a supported upstream memory path. The notebook's custom attention patch has not been independently verified; if indispensable, compare nonconstant predictions against unpatched inference at a declared, preserved tolerance before a full run.
- Fix a balanced number of completed units per fold. Its deadline scheduler can leave different seed counts by fold, calculate AUC on only covered rows, and average whatever test units finished. Require 100% OOF coverage and the identical declared fold/seed recipe for test output.
- Record checkpoint hash, package version, precision, device, source hash and context size. Source identity alone does not establish inference equivalence.

Do not immediately expand to ten seeds. A failed one-estimator arm does not establish that the entire family fails: the public four-estimator result supports a separate bounded follow-up if resources and paired error evidence justify it. Freeze that follow-up contract before running it. Likewise, route/teacher feature inputs are a separate later ablation, not bundled with the raw-feature baseline.

### 2. TabM: one distinct neural model, conditional on budget

Hypothesis: a parameter-sharing neural ensemble with numerical embeddings contributes errors different from the existing RealMLP. Use the maintained TabM package and one fixed configuration, with fold-local preprocessing and an inner training-only stopping split (or epochs fixed from that inner split). Outer validation labels cannot choose epochs. Use raw predictors initially; do not silently inherit the entire RealMLP target-encoding recipe. Compare the same fixed 80/20 blend against the unchanged incumbent.

Use piecewise-linear numerical embeddings as the first proposed view; preserve rating zero as a distinct value, and record any categorical treatment explicitly. This is a model-family experiment, not evidence that a larger feature list helps. The upstream training guidance optimizes the mean of individual member losses and averages probabilities for inference. Do not replace that with loss on the averaged prediction. Exact package/configuration pins remain to be finalized before authorization and launch.

### 3. Combine only members that earn a place

First retain fixed blends. If both new members contribute, preregister one combined blend and validate it; do not search dozens of weights on exposed OOF rows. If evidence warrants learned weights later, use a small L2-logistic combiner on model logits and compare it with fixed weights.

Strict outer evaluation must exclude the held-out fold's labels from every base model producing meta-training features as well as from combiner fitting. Merely splitting a pre-existing OOF matrix again does not establish this: its meta-training rows can come from base models trained on the outer evaluation fold. Nested base fitting or a suitably isolated evaluation set is required for a stronger claim, especially for nonlinear or neighbor-reading stackers. Label-fitted blend evaluation on reused OOF can be reported as development evidence but must not be called untouched or fully nested.

### 4. Conditional representation lead, not another default feature sweep

If the above still misses source-disagreement cases, test one categorical interaction view (for example rating-pair/equality patterns) in an existing model, only after checking it is absent from current encodings and prior rejected arms. The public pool includes rating-cross and equality-block models, but inclusion in a pool does not prove marginal value. A source-teacher residual feature alone is not new: this project already tested related teacher and auxiliary approaches.

## Proposed gate and resource envelope — not launch authorization

These are proposed caps and decision rules, not measured runtimes or a fully executable experiment contract. Training remains paused. Before launch, the owner must resume training; record available quota, execution hardware, exact versions/configuration, job identity and monitor owner. Submission requires separate authorization.

1. After resumption, cap the TabPFN compatibility/memory pilot at 30 minutes on one allocated GPU. Reduced-context checks establish feasibility only, not full-context quality. Stop on incompatible runtime, missing model access or unverified patch behavior.
2. Cap the first complete seed-42 fold at two hours on one GPU. Measure fit/predict memory and throughput. If the projected complete five-fold job plus test predictions exceeds ten GPU-hours, stop and revise the resource plan explicitly. No automatic 11.5-hour multi-seed scheduler. TabM gets a separate proposed five-GPU-hour cap, one configuration only.
3. A one-fold result is a screen, never promotion evidence. Before completing an arm, freeze all parameters and the 80/20 blend. Require all five folds and report pooled AUC, paired per-fold differences, and paired stratified-bootstrap uncertainty on saved predictions. Bootstrap intervals are conditional on fitted models and do not account for the full history of adaptive selection.
4. Proposed useful-signal gate: pooled gain at least 0.00005, at least four of five positive paired folds, and a positive lower endpoint of the paired 95% bootstrap interval. This is deliberately above Batch 7's tiny failed gain; it does not imply the 0.00089 public gap is closed. If uncertainty is unresolved, report inconclusive rather than declare success.
5. Confirm a passing frozen arm on seed 2026 against that seed's matched incumbent; require the same gate without retuning. This is robustness on reused rows, not pristine validation. Keep seed-specific OOF separate. A later independent external score is still necessary to establish a public improvement.
6. Stop an arm on incomplete artifacts, invalid provenance, failed replay or the resource cap. Preserve all negative results. A retained incremental improvement and attainment of public >0.96200 are separate milestones.

Before any later adaptation, reach only the needed caller, packager, prediction replay and test surfaces. Reuse `verify_results.py` conventions and the existing `{id, fold, target, prediction}` output schema. `blend_equal.py` only handles equal weights; it is not already an arbitrary 80/20 or learned-weight implementation. No new generic experiment framework is justified.

## Approaches to deprioritize

- More RealMLP epochs, applicability summaries, teacher-enhanced neural features and original auxiliary predictors without new evidence: already tested here.
- Another broad LightGBM/XGBoost hyperparameter sweep: lower priority than testing complementary families, given the historical tree correlation and small recent gains.
- Threshold tuning or a strictly monotone transform of one finished prediction vector: ROC AUC is unchanged when ordering and ties are unchanged. Combining differently transformed model outputs can change ordering, but then needs honest blend validation.
- Consensus pseudo-labels built from models trained on all competition labels: they can transmit outer-fold labels through test predictions. Do not use them in validation without regenerating the teacher inside the appropriate folds.
- Deleting hard labels, assuming a 3.8% random-noise ceiling, importing public predictions with unknown folds, or selecting rank extrapolations by repeated public submissions. These do not establish the requested reproducible improvement.

## Source provenance

Public notebook sources were fetched read-only through Kaggle's public kernel-pull endpoint on 2026-10-24 and parsed as JSON/text, never executed. Hash is SHA256 of the returned `blob.source` UTF-8 string. Copies remain under `/tmp/s6e10-research`, outside Git. Numeric API version and prose part numbers may differ.

| Public notebook | API version | Source SHA256 |
|---|---:|---|
| Hemingweb full context | 2 | `6f683e494124f1169a413bb38646b046d1e3362602d9893e3fb21ad95a013c82` |
| Goodpjw2008 twin stacks | 13 | `b4593352082d25e89e6d4a5c79de7b9e1cf5e11e02e359f848a62307f7cd60af` |
| Rabari public-surface blend | 55 | `7b52a284f439f7b90dc84c2ceefa7f7f672878bf3029c1b275c053ad27dc0cab` |
| Dariush label-noise analysis | 7 | `b4e4640b632bd572f23462fafe0fb73a8339f5ec912e10e454f10a51e9215f60` |
| Deotte foundation-model stack | 1 | `bedc1ccc11ede29ae1c663ca64c14d4af591e647bebac75ea2bdb7940e3c2fa4` |
| Busyaprime route reference | 1 | `4d3614bc2e42f414d560ad3d16646861e456eaffedc688c397c926c9b2b2021a` |

Hemingweb's source hash matches the earlier ledger snapshot, now with API version 2 identified. Some web-rendered notebook pages were inaccessible or stale; current public API source metadata was used where available. No prediction arrays, competition CSVs or checkpoints were downloaded. No private access was bypassed. The screenshot contains an authentication token, not a quota readout; the token was not used, stored, or reproduced in project files. Remaining GPU quota is unknown.

## Verification and handoff

- Reviewed source for existing neural/tree training, feature preparation, equal blending and verification; no runtime code changed.
- Checked six public notebook source hashes and version metadata; no notebook code executed and no reported scores independently recomputed.
- Documentation link, whitespace and credential-pattern checks are the appropriate verification for this documentation-only change; runtime tests are not required and were not run.
- Ownership claim was published as `87d6fc243fc28775e647b930502b09854021f7b9`; remote SHA readback matched. Final research publication is recorded by the subsequent Git commit and remote readback.
- Handoff: recover incumbent prediction artifacts, then finalize and authorize the bounded TabPFN contract if training is resumed. Current research does not prove any candidate beats the incumbent, and no >0.96200 result has been achieved here.
- No training, GPU smoke, remote job, prediction submission or credential installation occurred.
