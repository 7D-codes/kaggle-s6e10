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
