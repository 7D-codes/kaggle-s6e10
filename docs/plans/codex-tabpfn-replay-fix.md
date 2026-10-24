# Fix TabPFN batch-dependent replay

Owner: root Codex / gpt-6.1-sol. Branch: `agent/codex/bounded-tabpfn`. User explicitly requested “fix it” after the two failed pilot results. This authorizes fixing the demonstrated inference failure; the no-competition-submission instruction remains in effect. TabM work is deferred while this fix is investigated.

Current boundary: the owner's latest supplied AGENTS.md explicitly pauses training, GPU smokes and remote jobs, and invalidates past launch permission. This correction proceeds as software maintenance only. The proposed native verification below is conditional on explicit resumption; no new job will launch in this task.

## Evidence and root-cause investigation

- Autocast pilot kernel140608582/v1: repeated64-row queries passed;64versus1000query comparison failed28/64rows, max violating difference0.00048846, at unchanged `rtol=1e-6, atol=1e-7`.
- FP32 correction kernel140610219/v1 (`wasidyms/s6e10-codex-tabpfn-fp32-pilot`) also failed the cross-size comparison:6/64rows, max violating difference3.90410423e-6. Runtime154.130008667seconds, T4 peak allocated3312146944bytes; same frozen checkpoint/data/seed/context and original checks. Total used GPU quota0.0685361111111111hours. No complete candidate export, full-fold run or submission.
- Pinned upstreamv9 inference.py1187–1204 propagates forced dtype;1243–1248 explicitly documents query-chunk floating-point differences and links [official issue800](https://github.com/PriorLabs/TabPFN/issues/800). Cached architecture uses frozen train statistics/inducing states, context-only ICL/decoder attention and row-local column aggregation. No intentional query-query mixing was found.
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
5. Only after explicit owner resumption, root alone may launch one reviewed private stable-dispatch diagnostic/verification job, fixed **1500seconds** including setup/download/fitting/checks. Previous81.7185+154.1300+1500seconds stays below the original1800second combined pilot envelope. Server cap also1500seconds. No automatic retry, relaxed tolerance or search for a passing block size. This is a proposed contract, not current launch authorization.
6. Stop if any semantic/slot/permutation/replay check fails, or if resource/coverage/provenance fails. No full-fold training before numerical acceptance and a resource forecast. Passing this fix does not establish a score improvement.

## Progress

Source investigation and read-only review complete; implementation ownership will be handed to an isolated worktree. Root owns this plan, strategy and scientific ledger. No new job has launched for this fix.
