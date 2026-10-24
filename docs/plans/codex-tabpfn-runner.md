# Codex TabPFN runner implementation

Agent: Codex / GPT-6. Branch: `agent/codex/tabpfn-runner`. Root execution contract: [bounded TabPFN](codex-bounded-tabpfn.md).

## Scope

New runner, packager and two synthetic test files only; preserve existing runtime modules. Raw 21 features; seed-42 five-fold split, first 50,000 fold-0 context rows, first 1,000 validation/test rows. Fold-local categories, exact pinned parameters, safe fresh output and provenance. Repeated nonconstant prediction tolerance remains rtol=1e-6, atol=1e-7. This is feasibility, never full OOF.

Private Kaggle package embeds exact runner and uses a supervised child for setup/download/execution under one 1,800-second deadline. Root alone may launch. No credentials handled, actual downloads/training/GPU jobs/submissions run by this agent.

## Verification plan

Establish existing 48-test baseline in isolated dependency environment. Observe new synthetic tests RED before implementation. Test fold/context selection and unseen categories with real preprocessing; use minimal fake estimator only for wiring without TabPFN/torch. Test probability rejection, output non-overwrite, embedded source and private metadata, timeout cleanup and reporting. Run targeted then full suite in source-only checkout. Record actual outcomes and remote commit readback below.

## Progress

Claim prepared after origin fetch and parent-branch fast-forward; initial branch has no upstream yet. Dependency environment exists but pytest installation is not yet complete; baseline pending root environment preparation.

## Implementation checkpoint

- Baseline: 48 passed, 66 existing warnings, 13.22s using `/tmp/s6e10-test-env/bin/python -m pytest -q --ignore=test_tabpfn.py` before implementation.
- RED: missing runner import and missing packager fixtures reproduced; later targeted RED demonstrated missing device metadata in failed receipts. No real model fit, download, GPU smoke or submission occurred.
- GREEN: 20 targeted synthetic tests passed in 2.25s. First full suite passed 68 tests with 66 existing warnings in 13.79s; fresh source-only verification is pending after checkpoint.
- New files reach only their own CLI/packager/tests; existing callers/config/exports remain unchanged. Packager pins official checkpoint revision `b0662010dd17679509f42326e2f9ba37a6f58818` and SHA256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3` supplied by root's independent official API check.
- API checked against official [TabPFN v9.0.0 classifier source](https://raw.githubusercontent.com/PriorLabs/TabPFN/v9.0.0/src/tabpfn/classifier.py). No custom patch, fallback, hyperparameter flexibility, runtime dependency additions or submission path.
- Pilot receipt includes context/validation/test row counts, exact model parameters, category vocabularies, versions, source/checkpoint/input SHA256, AUC explicitly labeled pilot-only, device/peak memory, runtime and submissions=0. Repeated 64-row nonconstant predictions must match one another and the original batch at unchanged rtol=1e-6/atol=1e-7.
- Supervisor starts its deadline at script startup; installation/download and runner all execute in a separate process group. Deadline expiry kills descendants and records stage, return code and status. No retry. Package generation itself refuses overwrite.
- Limits: synthetic estimator verifies wiring only; actual TabPFN/CUDA memory feasibility and native numerical replay remain unverified until root's separately owned authorized pilot.

## Root review follow-up

- Added exact competition train/test SHA256 verification before package setup, download or fitting, using hashes independently verified by root. Drift stops execution; no substitute input is accepted.
- RED reproduced two issues before correction: missing worker input-hash verification, and malformed `stage.json` raising during failure-receipt handling. GREEN: 22 targeted tests passed in 2.23s.
- Fresh source-only archive at implementation checkpoint `eceae2c1838a6ac0ead98b8070d74d559099c969`: 68 passed, 66 existing warnings, 12.70s. Latest hash/receipt follow-up requires final fresh-archive rerun.

## Handoff

- Final code checkpoint `82b586cf04ebcb00c7d590c058e49f50d3160907` published and remote SHA matched. Fresh `git archive HEAD` source-only checkout `/tmp/s6e10-tabpfn-final-source-73xg1mdv`: **70 passed, 66 existing warnings, 13.66s** with `/tmp/s6e10-test-env/bin/python -m pytest -q`. No private inputs, torch or TabPFN needed for verification.
- Whitespace/byte-compilation checks passed; staged filenames and credential-pattern matches inspected. Only the four new source/test files and own coordination evidence changed; no restricted artifacts were staged.
- Implementation is ready for root's ordered specification/quality reviews and integration. Ownership of the four runtime/test files is handed back to root after this checkpoint; root alone owns any authorized pilot launch/monitoring. Root may request follow-up edits through explicit handoff.
- No data/model download, real model fit, GPU smoke, remote job or submission occurred here. Real TabPFN 9.0.0 installation, native numerical replay and 50k-context T4 memory/runtime feasibility remain unverified; the pilot must stop with its receipt if any gate fails. Pilot AUC cannot be used as full OOF or promotion evidence.

## Metadata integration follow-up claim

Root explicitly handed back only the package title/ID consistency correction, existing metadata regression test and this task evidence. Root reports the first authorized pilot was canonicalized to `wasidyms/s6e10-codex-tabpfn-feasibility-pilot`, version 1, kernel `140608582`, because the title contained an extra “Feasibility” token. Root has verified that running pilot's source and privacy and retains sole monitoring ownership. No duplicate launch or restart is needed; competition submissions remain forbidden. Plan: reproduce title-to-ID mismatch with the existing metadata test, change the future package title to `S6E10 Codex TabPFN Pilot`, run targeted checks and publish. No launch, runtime-source or root-owned document changes.

Follow-up verification: the existing metadata test first failed with normalized title slug `s6e10-codex-tabpfn-feasibility-pilot` versus intended ID slug `s6e10-codex-tabpfn-pilot`. Changed only the metadata title value to `S6E10 Codex TabPFN Pilot` and added one title-slug assertion to that test. Targeted packager suite: **6 passed in 1.05s**; whitespace check passed. Generated runner/worker source is unchanged. Historical running-kernel identity remains root-owned evidence; this correction applies only to future package generation. No launch, restart or submission occurred. No full-suite repeat for this title-only change; prior 70-test source-only evidence remains above.

## Fixed FP32 correction arm claim

Root reports the authorized autocast pilot `wasidyms/s6e10-codex-tabpfn-feasibility-pilot`, version 1, kernel `140608582`, failed native 64-versus-1,000-query batch replay: 28/64 differences, maximum absolute error 0.00048846. Same-64-query replay passed. Reported total runtime 81.7185s, GPU quota 0.0241033h, Tesla T4 peak allocation 2,490,343,936 bytes; runtime Python 3.13.15, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, TabPFN 9.0.0, torch 2.11+cu128. This is a failed numerical gate, not a software unit-test failure; root owns the canonical historical evidence.

By explicit root handoff, this agent owns only the existing runner/packager/tests and own evidence for a separate fixed FP32 correction arm. All data, split seed 42, first 50,000 fold-0 context rows, model seed/checkpoint, categories, int8 KV cache and other model parameters stay fixed. Serialized precision becomes `float32`; lazy backend construction converts that one fixed value to `torch.float32`. Both nonconstant replay checks remain at rtol=1e-6/atol=1e-7. No tolerance relaxation, deleted check, custom patch, automatic retry or general precision option.

Future private kernel ID/title: `wasidyms/s6e10-codex-tabpfn-fp32-pilot` / `S6E10 Codex TabPFN FP32 Pilot`. Fixed total budget: 1,710 seconds including input verification, setup, download and inference, leaving original rounded 82s + 1,710s = 1,792s below the aggregate 1,800s cap. Root alone will review, publish the new prelaunch contract and decide the separate launch. No jobs, model/data downloads or submissions by this agent. Plan: focused RED/GREEN backend-dtype and metadata/cap tests, then full fresh source-only suite and pushed handoff.
