# Strategic reassessment — 2026-10-27

Owner asks to step back and assess how to obtain a better score. Training and the persistent goal remain paused; no fitting, searches, GPU smokes, launches or submissions. Root Codex owns this document; all advisors are read-only.

## Question and evidence

What is the highest-value next strategy for improving the final ROC AUC above the public Summit anchor, given the existing artifacts and previous failures? The retained independently submitted public score is 0.96111; supplied Summit source reports 0.96213 and exact OOF is unavailable. CatBoost added to a frozen eleven-member surrogate yielded development pair-weighted AUC 0.96168694246 to 0.96179190793, five positive folds. Export uses rank(rank(Summit)+0.05*(candidate probability-control probability)); improvement of that final artifact is unknown. Twelve-member control already contains nine public TFMs, fixed-epoch own RealMLP, original-only XGB teacher and own CatBoost. Public producer provenance is conditional; historical development rows are exposed. Original-data priors were already explored; longer neural fitting and several subgroup/teacher blends were rejected.

## Scope and method

Read current ledger and source evidence. Five independent perspectives, anonymous peer review and synthesis. Inspect saved CSV/JSON only, verify hashes and evaluate the already-fixed correction on its own matched OOF control; no fitting or coefficient sweep. Quantify exported test-rank movement without attributing test quality. Prefer a short prioritized queue with unchanged controls, credible validation and stop rules. No runtime source changes or new dependency.

## Progress and verification

Repository synchronized at cbf865ed3cdfa94c01621a9dd8f790470e4d18c3. Own interrupted handoff corrected to distinguish a stalled local observer from confirmed remote execution. Findings pending.

## New measured finding: the fixed export attenuates the gain

Artifact-only diagnostic, no model fitting, weight sweep, GPU action or submission. The already-fixed intervention was evaluated separately inside each of the ten original fold/half cells:

`adjusted = percentile_rank(percentile_rank(control) + 0.05 * (candidate_probability - control_probability))`

AUC is computed per cell and combined with positive-negative pair-count weights, exactly matching the historical primary aggregation. This substitutes the matched surrogate control for the unavailable Summit OOF anchor; it does not estimate Summit improvement.

| Saved-prediction comparison | Pair-weighted development ROC AUC | Gain over matched control |
|---|---:|---:|
| Eleven-member control | 0.9616869424637373 | — |
| Full twelve-member CatBoost candidate | 0.9617919079329712 | 0.00010496546923383286 |
| Existing fixed correction applied to matched control | 0.9616938478874067 | 0.000006905423669447574 |

The export analogue retained **6.57875749%** of the full candidate's measured gain. All five original folds remained positive: 0.00000634987, 0.00000764275, 0.00000978158, 0.00000507103, 0.00000568188. This positive but much smaller exposed-row effect diagnoses attenuation. It does not establish that 0.05 is mathematically wrong, that a larger coefficient is safe on Summit, or that the existing inexpensive artifact should be deleted. The conservative coefficient may limit uncertain transfer risk. No coefficient was selected from these results, and no historical promotion gate was retroactively changed.

On the actual saved test files, the exported candidate exactly replays the formula (maximum absolute error zero). Against Summit its median absolute rank displacement is 20 positions among 299844 rows; p90=97, p99=460, maximum=2396. Spearman correlation is 0.9999993232715467. These describe movement only, not prediction quality.

### Verification and provenance

All five artifact hashes in the accepted comparison metrics matched: OOF, test, unchanged Summit CSV, candidate CSV and coefficients JSON. Full row counts, unique/aligned IDs, finite predictions and original fold/cell consistency checked. Diagnostic reproduction of historical primary AUC/gain passed to 1e-14; the existing export replay tolerance remained rtol=1e-6, atol=1e-7 and actual error was zero. Nonconstant synthetic identity and tied-rank checks passed. Initial one-off script syntax error was corrected before execution; no partial metric output was accepted. No model objects were loaded.

- OOF CSV SHA256: `81a03f255444afd113b76f45ece17e978be4b517410dec61a72b6930502eebf3`.
- Test CSV SHA256: `2fd72ff51b930151578592b4c17596bb5256deb646f5debfcd211ec52dc711bd`.
- Summit SHA256: `0c6d65f427725bd57ed2885826c451f6eb0952a983bd62ab8a397763b60475c0`.
- Existing candidate SHA256: `93282e7daa51ffa02f411fa186324edbb83143df49790933a6415087721cbd6b`.
- Diagnostic source SHA256: `941d6f9b57530cc18c7cda985ba25c938d14a3e14de17b1047a5775b2de59b39`.
- Diagnostic JSON SHA256: `8dedbaf634e28076ed534f735c9d60b46c03d0db40e12cf106f18513a5eaf952`.
- Local ignored source/report: `research/strategic-reassessment-20261027/fixed_correction_diagnostic.py` and `fixed-correction-diagnostic.json`; inputs remain in the preserved bounded-tabpfn worktree. Public-safe method and results are recorded here for teammates without those files.
- Runtime actually checked: Python3.13.11, NumPy2.5.4, pandas3.1.0, SciPy1.18.1, sklearn1.9.1. Original fivefold seed42, meta half seed20261025+fold, ten conditional evaluation cells. No new random splits or bootstrap claims.

## Revised priorities after training is explicitly resumed

1. **Evaluate the final recipe using existing predictions first.** Preserve Summit as a fixed external anchor and the full twelve-member candidate as an independently evaluable challenger. The present incremental gain justifies retaining that challenger, but it does not validate the small Summit overlay. Reuse existing fold-compatible predictions to compare the unchanged combiner with one predeclared alternative that evaluates the complete emitted ranking. Fit any coefficients/selection inside the meta-training partition; keep the held-out partition out of every fitted stage. Scale choice and final blending must be part of the evaluated intervention. Do not simply multiply 0.05 after seeing this diagnostic, and do not spend GPU quota to conduct a CPU combiner comparison. No exact Summit OOF means no offline claim of superiority to Summit; its unavailable recovery must not become another open-ended project.
2. **Close the already-launched Kumo experiment once its receipts are available.** Its raw/engineered comparison is only eligible with full native/provenance/coverage checks and the frozen gates. After authorized evaluation, judge the final exported recipe as well as full-stack contribution. A failed resource/quality contract closes this attempt; no automatic retry or another infrastructure council. The bounded read-only follow-up below refreshed its completed status and fetched summary receipts; it did not cancel the job or run its prepared comparison.
3. **If more signal is needed, prioritize one TabM challenger with numerical embeddings.** The named twelve-member inventory has no TabM member. Hypothesis: supervised numerical representation and a jointly trained neural ensemble contribute ranking information beyond fixed-epoch RealMLP and the existing TFM pool. This is an unproven, bounded hypothesis, not guaranteed diversity. Use one raw-feature configuration, fold-local preprocessing and inner-only stopping, same control/folds and a fully declared final combination. The [official TabM implementation](https://github.com/yandex-research/tabm) supports numerical embeddings and parameter-sharing ensembles; it does not establish an S6E10 gain. Full-context TabPFN remains a fallback only if its checkpoint/context materially differ from our existing TabPFN member and a complete run is feasible; it is not the default merely because it is a strong model family.

A proposed economic target for the **next** compute-expansion decision is at least +0.00010 development AUC in the actual evaluated final recipe, with at least four of five positive folds, a predeclared paired uncertainty check and one robustness repeat. This is a prioritization target, not a promised public improvement, a new verdict on historical artifacts, or a complete launch contract. Freeze exact parameters, resource cap and stopping rule before resumption execution. A cheap positive artifact can be retained even when its effect is too small to justify more GPU spending. Repeated splits on already-exposed rows remain robustness evidence; stronger generalization claims require genuinely new labeled evaluation or a separately authorized competition comparison.

## What to stop repeating

Do not equate standalone strength, more estimators, a different model name or spare hardware with incremental information. The current control already contains nine public TFMs, including Kumo and TabPFN. Generic service summaries, label-defined specialists, longer neural schedules and teacher additions already have negative or weak evidence; a new trial needs a specific different hypothesis. Public leaderboard-selected blend files and high CV titles do not establish a reproducible advance. A fresh leaderboard page could not be read through the web tool in this assessment, so no current rank or top-score claim is made.

No runtime callers, tests, fixtures, configs or exports were changed. Runtime regression suites were not rerun for this documentation/artifact-only assessment. Existing datasets, models, predictions and job artifacts are preserved. The persistent goal was checked and remains PAUSED.

## Independent council and decision

Completed five independent standard-model advisors (Contrarian, First Principles, Expansionist, Outsider, Executor), followed by five anonymous peer reviews and a standard-model chairman. No additional Astra expert call. Four-agent concurrency limit required waves; later advisors did not receive earlier answers. Root prepared anonymized substantive response summaries and randomized A–E ordering; peer reviewers did not receive the identity mapping. Chairman received all five summaries/identity mapping, all five review summaries and the actual diagnostic. This is advisory agreement, not independent statistical confirmation.

Agreement: compare the delivered ranking, use already-paid-for predictions, preserve inexpensive positive artifacts, and stop equating surrogate gains with Summit gains. Disagreement: how much effort to spend recovering Summit OOF and whether full-context TabPFN or a new representation is the next best model. Peer review rejected indefinite provenance recovery, automatic coefficient increases, retroactive failure declarations and interpreting a new split as untouched validation. It separated three decisions: retaining a cheap artifact, spending new GPU time, and submitting.

Chairman recommended closing the existing Kumo receipt, evaluating a reproducible full challenger separately from the conservative overlay, then one bounded TabM hypothesis if more training is justified. Root adopts that strategy; obtaining a status receipt does not authorize model fitting. Current saved-prediction diagnostic is complete, so it need not become another monitoring/research project.

Local advisory packet hashes (summaries, not verbatim transcripts): anonymous responses `4e6ba7ad3e52a1ceaf357a9a588b32e34903cb6391f72749378923f55611c377`; five peer summaries `13bd1297892c87e0f7d5c1e2853a9c97b0ac7e85d8b495487f2fec5b55703762`. Team-readable reasoning and results are contained in this document; private predictions are not required to understand the decision.


## Final read-only Kumo status receipt

At **2026-10-27T05:35:55.601227 UTC**, the API for exact `wasidyms/s6e10-codex-kumo-continuation-v1/1` reported **COMPLETE**, empty failure message. One bounded read-only call fetched small JSON summaries only; no restart or real-data comparison. Worker receipt reports returncode0, 18202.56696 seconds total runtime against24900 cap, full five-fold OOF and 210/210 remote native comparisons passing. First pair3697.63975seconds/projectedtotal23110.24845 satisfied the frozen projection. These are remote self-reports, not the independent full-artifact audit; quota attribution also remains unchecked.

Reported standalone OOF AUC is raw **0.9574360974220885**, engineered **0.9584955413824405** (difference **0.0010594439603520**). This is a within-Kumo standalone change, not incremental contribution to the twelve-member ensemble or Summit; no promotion or rejection follows from these scores alone. The main next evaluation can use the completed predictions without another GPU training run, after explicit resumption.

Execution summary SHA256 `57bdd3a008880a930061c310d3299ffcdcbe079dc6536c66d998499d6b5f7013`; metrics summary `cc3e7ad59f0abb2c09ee82e161fa253e43948aeb324f253645465bc3979d94d5`. Local ignored receipt folder: `research/strategic-reassessment-20261027/kumo-summary/`. Runner hash06d2591c… matches the frozen source identity; a source hash is not inference equivalence. Model KumoSmall/twoestimators/fp16/devicecuda:0, fixed100kcontext/fold, sourcefdc8372b…/checkpoint3c3e10bb…, fivefoldseed42. Summary runtime: Python3.13.15/NumPy2.1.3/pandas2.3.3/sklearn1.6.1/torch2.11.0+cu128. Full CSV/hash/context/native preprocessing audit and frozen ensemble comparison remain outstanding.
