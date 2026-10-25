# Existing CatBoost job: read-only audit

Owner: Codex / GPT-6. Branch: `main` for documentation. Runtime and artifacts remain on historical `agent/codex/bounded-tabpfn`; no runtime source changes planned. Training is paused under the latest owner instruction. No launch, restart, GPU smoke, search, score-improvement fit or submission is authorized by this audit.

## Objective and evidence

Resolve the actual state and completeness of existing private job `wasidyms/s6e10-codex-catboost-ctr-v1/1`, kernel 140747734, launched 2026-10-25 08:54:54 UTC. The preceding council checked it RUNNING at 09:52:20 UTC, empty failure message. Last turn produced verified wait evidence and a published advisory decision, not score improvement.

## Approach and scope

1. Poll this exact job; an observer timeout is not terminal and never permits restart.
2. On authoritative terminal state, preserve authorized own outputs and inspect JSON/CSV/hash evidence. Verify execution status, five-fold coverage, pinned configuration/source/runtime, native checks, exact IDs/targets/folds and matching output hashes. Do not load untrusted pickle/joblib.
3. Preserve incomplete/negative evidence. No partial-OOF promotion and no comparison fitting during the pause. Keep later frozen scoring contract unchanged.
4. Record actual verification and next handoff here. Only this plan and Codex's audit registry subsection are owned; historical runtime files and other agents' entries remain untouched.

## Progress and handoff

Read-only audit COMPLETE; ownership released. Claim was published before audit work. Existing job and independent artifact/native audit passed. Any new training or score-improvement fitting requires explicit resumption; submission remains forbidden.

## Checkpoint: 2026-10-25 09:57 UTC

- Claim commit `4216d2f5dfd07d8ef3eda60b20599a68bb8f5b27` matched remote main readback.
- Exact-job API polls at 09:54:18, 09:55:53 and 09:57:21 UTC all returned `KernelWorkerStatus.RUNNING`, empty failure message. This is a verified live wait, not a terminal failure or permission to restart.
- Available live-log snapshot contained zero characters. No terminal output receipt/directory exists locally; fold progress, actual execution start, acceptance and predictive quality cannot be inferred from elapsed launch wall time.
- Inspected the frozen runner's receipt structure: outer/inner index hashes, selected iteration count, GPU runtime parameters, native replay checks, completed-fold list and final output hashes must be audited on terminal outputs. Full OOF/test requires all five folds; native reference tolerance remains rtol=1e-6/atol=1e-7.
- No executable source changed; no models loaded, new fits, job mutations or submissions. Documentation whitespace, links and credential-pattern checks are the relevant validation; runtime tests are not needed.
- Handoff: keep monitoring this same version. If terminal, preserve and audit outputs before any predictive conclusion. Training pause remains in force and the score objective remains unfulfilled.

## Terminal result and independent verification — 2026-10-25

- At 09:59:34 UTC the same-version API returned COMPLETE, empty failure message. Execution receipt: return code 0, 3798.0025 seconds total, no deadline exhausted, zero submissions. First fold 856.6035 seconds; all five folds complete. No restart or job mutation occurred.
- Preserved 49 authorized own output files (approximately 1.8 GiB) locally. Their metadata/source/configuration and model-file hashes were audited, not merely trusted from API completion. Official input hashes, exact IDs/targets, unique full coverage, seed-42 outer folds and all inner split index hashes match.
- Native training reports CatBoost 1.2.8, NumPy 2.1.3, pandas 2.3.3, sklearn 1.6.1 and Python 3.13.15. Source/parameters/features match the sealed package. Configuration SHA256 `1993d0b75aecef23a52d2cb11c8efffecf871c7de89a93297ac0b544be89d785`; selected trees by fold: 3655, 2962, 2311, 3726, 3653. Fresh outer refits have no early-stopping settings, and inner fits use only their declared inner holdout.
- All 60 recorded native checks pass at unchanged rtol=1e-6, atol=1e-7. Independently reloaded hash-verified own CBM models and replayed 9965 validation rows (first 1000 plus a separate fixed random sample per fold, deduplicated): maximum absolute error zero in every fold. Checked categorical vocabularies against fitting rows. This is native component spot replay with frozen qualified teacher inputs, not full-test raw-row teacher regeneration.
- Independently matched per-fold validation files to OOF, averaged all five full test vectors, and recomputed standalone development ROC AUC **0.9610177594499577**. This is not a public score, ensemble gain or comparison against exact Summit OOF.
- Final OOF: 699635 rows, SHA256 `17765f1e736d17cc80444d5654b928e0f20e43a8d8acf3ec03a96fc6cf40530b`. Test: 299844 rows, SHA256 `46df1eebea22434b09a320b75a916e67a1942e439800ec354c1db8808f149c50`.
- Metrics SHA256 `5ded31790d140a752691794185b81ad0afdce81cb73c3b28f7882095ecc35dab`; execution SHA256 `19182afbe6eb70b0a7629b161641ddebf11bcfbc6e4a91f2611dc18f071f8b44`. Local audit source `research/summit-score-goal/analysis/audit_catboost.py`, SHA256 `b3c7ca43aeff507918780a5455150995d834f2f7ea1798b0355129e119e79421`; invalid/constant prediction self-check passed. Local receipt: `reports/summit-score-goal/catboost-execution/independent-terminal-audit.json`, valid/full OOF/full test true. These ignored paths are in the historical bounded-tabpfn worktree, not fresh-clone inputs.
- Quota read at 10:03:02 UTC: 2.1747102778 GPU hours used, **27.8252897222 remaining**. This is a timestamped provider snapshot, not a promise of future availability.

## Decision and remaining work

The Cat component is integrity-qualified. Council's later Kumo control should include it regardless of its future score-gate result. Its frozen eleven-versus-twelve-member comparison remains unexecuted because it requires new logistic fits during the current pause. Asked the owner whether to resume comparison and capped experiments, comparison only, or keep audit-only scope; no answer at this checkpoint. No dependent fitting proceeds without the answer.

No promoted Summit correction or verified public improvement exists. The original Summit bytes and all negative/partial Kumo evidence remain preserved. On explicit resumption, run the existing frozen comparison once; do not tune depths, features, weights or C after the outcome. No competition submission.

Documentation-only tracked changes: local-link, whitespace and credential-pattern checks passed. Runtime source was unchanged, so no full runtime suite was rerun. Final publication and remote readback are recorded in Git history.

## Score-goal blocker audit

The preceding goal turn made progress by completing the terminal/native/coverage audit. The training-pause condition has persisted across at least three consecutive goal turns; permitted read-only checks are now complete. Refetched main and reread its explicit pause: no owner resumption has arrived, and the asynchronous clarification remains pending. The historical goal's permission does not override the later instruction that past job permission is not reusable.

Revalidated the independent receipt as valid/full OOF/full test and the prepared comparison source against frozen SHA256 `5928390764f92c5e4e381975620290ae341ca0358cf6654e3a27747f855591b9`. No comparison output directory exists; no comparison fits have run. The next concrete action is the unchanged eleven-versus-twelve-member comparison, requiring new logistic fitting and therefore explicit scope resumption.

Also inspected preserved Kumo dispatch receipts read-only: raw completed with 3475 canonical calls, FE snapshot had 1132 calls and was in test inference. They contain no separate arm/inference duration measurements. A conservative complete-work forecast cannot be claimed from those counters; it needs the proposed bounded first-complete-pair experiment, also forbidden during the pause. No new source/control/model search is justified to bypass this boundary.

Score-improvement work is BLOCKED pending owner resumption; the requested higher score is not achieved. All useful artifacts, source snapshots and frozen gates remain preserved. No live job remains from the audited Cat run, and no new job, fit or submission occurred during this blocker audit. On resumption, use the same audited artifacts and frozen comparison; do not retrain or duplicate CatBoost.
