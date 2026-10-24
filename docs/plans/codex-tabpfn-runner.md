# Codex TabPFN runner implementation

Agent: Codex / GPT-6. Branch: `agent/codex/tabpfn-runner`. Root execution contract: [bounded TabPFN](codex-bounded-tabpfn.md).

## Scope

New runner, packager and two synthetic test files only; preserve existing runtime modules. Raw 21 features; seed-42 five-fold split, first 50,000 fold-0 context rows, first 1,000 validation/test rows. Fold-local categories, exact pinned parameters, safe fresh output and provenance. Repeated nonconstant prediction tolerance remains rtol=1e-6, atol=1e-7. This is feasibility, never full OOF.

Private Kaggle package embeds exact runner and uses a supervised child for setup/download/execution under one 1,800-second deadline. Root alone may launch. No credentials handled, actual downloads/training/GPU jobs/submissions run by this agent.

## Verification plan

Establish existing 48-test baseline in isolated dependency environment. Observe new synthetic tests RED before implementation. Test fold/context selection and unseen categories with real preprocessing; use minimal fake estimator only for wiring without TabPFN/torch. Test probability rejection, output non-overwrite, embedded source and private metadata, timeout cleanup and reporting. Run targeted then full suite in source-only checkout. Record actual outcomes and remote commit readback below.

## Progress

Claim prepared after origin fetch and parent-branch fast-forward; initial branch has no upstream yet. Dependency environment exists but pytest installation is not yet complete; baseline pending root environment preparation.
