# Fix TabPFN batch-dependent replay

Owner: root Codex / gpt-6.1-sol. Branch: `agent/codex/bounded-tabpfn`. User explicitly requested “fix it” after the two failed pilot results. This authorizes fixing the demonstrated inference failure; the no-competition-submission instruction remains in effect. TabM work is deferred while this fix is investigated.

Current boundary: after the software-only correction, the owner explicitly resumed the capped GPU verification: “connect using the kaggle info in the image and push to the gpu, check qouta first.” This permits exactly one reviewed Stable Pilot after quota/active-job checks, with root as launch owner/monitor. Full-fold training/search remains paused and submissions prohibited.

## Evidence and root-cause investigation

- Autocast pilot kernel140608582/v1: repeated64-row queries passed;64versus1000query comparison failed28/64rows, max violating difference0.00048846, at unchanged `rtol=1e-6, atol=1e-7`.
- FP32 correction kernel140610219/v1 (`wasidyms/s6e10-codex-tabpfn-fp32-pilot`) also failed the cross-size comparison:6/64rows, max violating difference3.90410423e-6. Runtime154.130008667seconds, T4 peak allocated3312146944bytes; same frozen checkpoint/data/seed/context and original checks. Total used GPU quota0.0685361111111111hours. No complete candidate export, full-fold run or submission.
- Pinned upstream v9 inference.py1187–1204 propagates forced dtype;1243–1248 explicitly documents query-chunk floating-point differences and links [official issue800](https://github.com/PriorLabs/TabPFN/issues/800). Cached architecture uses frozen train statistics/inducing states, context-only ICL/decoder attention and row-local column aggregation. No intentional query-query mixing was found.
- Upstream's memory-saving helper flattens leading dimensions and chooses split sizes from total row count. GEMM/SDPA shapes therefore change between64and1000queries. Repeated same-shape success and cross-shape failure fit documented numerical non-associativity. This is strong causal evidence; it does not identify our first differing low-level CUDA operation.
- Read-only independent root-cause review agreed that a canonical query-dispatch recipe is principled if native permutation, slot and padding tests pass. Float64 is an upstream-supported reference alternative, but its T4 memory/runtime cost at50000context rows is unverified; do not launch another blind precision change.

## Minimal correction and scope

Reach every candidate generation/replay call in `train_tabpfn.py`, its synthetic tests, and the private pilot packager/metadata/tests. Preserve all model parameters, folds, context, preprocessing, checkpoint hashes and replay tolerances. No changes to incumbent modules, predictions or retained scores.

Define one canonical inference helper: split incoming queries into blocks of exactly **128rows**; repeat the final real query to pad a short tail; call the unmodified native classifier on each full block; discard padding and preserve original order. Use it for validation/test generation and every subset/replay path. Do not apply it solely to assertions. Record block size/tail policy and actual backend call sizes in the inference receipt. No batch-size search or new CLI option.

This changes the declared numerical execution recipe. Acceptance means reproducibility of that recipe, not equivalence to the already rejected variable-batch outputs. Padding is mathematically neutral only because this fitted cached model's query rows are independent; native tests must verify its actual preprocessing and arithmetic obey the claimed contract.

## Predeclared verification and remaining pilot budget

1. Synthetic TDD: reproduce shape-dependent probabilities; require fixed128calls, tail/order preservation and arbitrary input sizes; reject malformed probabilities and empty input; retain existing batch-drift rejection and non-overwrite checks. Test checkpoint-source-only operation without CUDA/models.
2. Keep both original64-row repeat and1000-versus64assertions unchanged at `rtol=1e-6, atol=1e-7`, routed through the production helper.
3. Native exact128real-row adapter/direct-classifier anchor; nonprefix sizes1,17,63,65,127,128,129; random permutation/inversion of1000rows; two predeclared rows repeated128times to test position/composition neutrality against their original mixed-block predictions. Original probabilities must remain finite, bounded, normalized and nonconstant. Record each comparison before asserting so failures remain evidence.
4. Record active fitted preprocessor configuration plus transformed-query equality/composition checks where exposed through the supported fitted estimator, without loading foreign serialized objects. Measure complete1000-row adapter runtime, actualT4 peak memory and code/config/input/checkpoint hashes.
5. Root alone launches one reviewed private stable-dispatch diagnostic/verification job, fixed **1500seconds** including setup/download/fitting/checks. Previous81.7185+154.1300+1500seconds stays below the original1800second combined pilot envelope. Server cap also1500seconds. No automatic retry, relaxed tolerance or search for a passing block size. Latest owner instruction explicitly authorizes this pilot only.
6. Stop if any semantic/slot/permutation/replay check fails, or if resource/coverage/provenance fails. No full-fold training before numerical acceptance and a resource forecast. Passing this fix does not establish a score improvement.

## Progress

Software correction integrated; native numerical acceptance remains pending. Root owns this plan, strategy and scientific ledger. No new job has launched for this fix.

## Software verification and handoff

- Isolated implementation branch `agent/codex/tabpfn-replay-fix`: code 336d971, evidence checkpoint e973859; explicit ownership handed back before root integration 528072fd25924f6cb29d3893efb5b3ee510270fd.
- Independent specification review PASS (36 targeted tests, 2.21s); independent quality review APPROVE (36 targeted tests, 2.50s), including inspection of the pinned upstream v9 cached member interfaces. Synthetic tests reject per-call drift, slot/composition dependence, malformed probabilities and missing cache mode; preserve padding/order and original tolerances.
- Fresh source-only Git archive of integrated 528072f at local `/tmp/s6e10-integrated-replay-lqh_ojju`: `/tmp/s6e10-test-env/bin/python -m pytest -q` passed **84 tests**, 66 existing warnings, 13.63s. No ignored competition data, model weights, Torch or GPU required. Same software-test environment recorded in the execution plan; this is not native inference proof.
- Packager generated and compiled the private Stable Pilot locally in that source-only checkout. Runner SHA256 `6880f4798d6f44d50386e763c424f0824cd2dc9ddb6eeb95e341ef6affee618a`; packager `a2e4c84a949736fee66ba98de07c0cbb856a71d20c5c592b365310c76ef01825`; generated source `6cd9ffe7ed8bff8f015623aaaf842a34e31f33c7447dba182b482585a31b5fa9`; metadata `b9f2baa52c401b0ac2492a020652dc9930d19b81373518de3ffa0774e6d96adf`. Packaging does not launch, install dependencies, download a model or submit.
- Every production/replay probability call now reaches the asserted128-row native boundary. Dispatch policy/configuration hash, actual call shapes/counts and calibration failures are retained in metrics. Direct17 CPU-transform comparison is diagnostic; padded/repeated128 CPU checks gate exact equality. CPU diagnostics inspect exposed fitted members directly; complete native probability gates cover classifier cleaning and GPU preprocessing when eventually run.
- Native T4 slot/permutation/padding/replay tests, runtime forecast, full OOF, paired gain and seed2026confirmation remain **unverified**. Arbitrary variable-size native outputs are explicitly not accepted. Do not promote this software change as a higher-scoring model.
- Current owner training pause prevents the proposed1500second verification launch. Prior two rejected runs/evidence and incumbent remain preserved; no third job, new model training or competition submission occurred during this fix. Draft integration [PR#2](https://github.com/7D-codes/kaggle-s6e10/pull/2) remains unmerged.

## Explicitly resumed native pilot

Owner authorization supersedes the software-only closeout boundary above for one pilot. Hypothesis/control/model/seed/data/features/checkpoint/FP32 and fixed128 recipe are unchanged from the reviewed contract. Acceptance requires every predeclared native/CPU gate, complete receipts and valid nonconstant1000-row validation/test predictions. No score-based tuning or full OOF decision from the pilot. Quota and own-kernel inventory must be checked before launch; require at least one remaining GPU hour and no active job. Preserve the previous generated FP32 package, generate the reviewed Stable package, verify all four hashes above, publish this contract, then launch once. Record returned version/source/privacy/status and stop on any failure. No submission.

Preflight at2026-10-24T19:45:26UTC: GPU used0.0685361111111111/30hours, remaining **29.93146388888889hours**, quota refresh2026-10-31T00:00:00. All12own kernels checked; no active job, both prior pilots ERROR, no existing Stable Pilot. Recreated package matches the four reviewed hashes above exactly; prior FP32 generated package preserved locally. Planned private id/title `wasidyms/s6e10-codex-tabpfn-stable-pilot` / `S6E10 Codex TabPFN Stable Pilot`, one visible T4, internal/server caps1500seconds. Root is sole launch owner/monitor; returned version pending. Local preflight/launch receipts under ignored `reports/tabpfn-stable-pilot/`.
