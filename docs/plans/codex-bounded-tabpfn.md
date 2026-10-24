# Codex: bounded TabPFN plan execution

Owner: Codex / gpt-6.1-sol. Branch: `agent/codex/bounded-tabpfn`. Worktree: `.worktrees/codex-tabpfn`. The owner explicitly authorized execution of [the research plan](codex-breakthrough-research.md). Training/GPU pilots in its caps are resumed; competition submissions remain unauthorized.

## Scope and order

1. Recover exact seed-42/2026 incumbent predictions and competition data using authorized Kaggle access; validate public-safe hashes and fold/ID/target alignment.
2. Verify checkpoint access and GPU feasibility. Do not assume an API token grants gated weight access or available GPU quota.
3. Implement only the required fold-local raw-feature TabPFN runner, private Kaggle packager and fixed-candidate comparison with synthetic TDD tests; preserve existing modules and inference tolerances.
4. Freeze the prelaunch contract and publish code/provenance before a capped pilot. Pilot <=30 minutes on one GPU; first full fold <=2 GPU-hours; projected full seed-42 run including test predictions <=10 GPU-hours. Stop and report resource/access failures.
5. Evaluate the fixed `0.8 * incumbent + 0.2 * candidate` blend, full coverage only. Gate: pooled ROC AUC gain >=0.00005, >=4/5 positive paired folds, paired stratified-bootstrap 95% lower endpoint >0. Confirm passing frozen recipe on seed 2026 with the same gate. Reused-row validation remains development evidence.
6. Conditional TabM stage is one configuration, <=5 GPU-hours, after resource/access requirements are resolved. Learned combination requires a separate validation design within the approved research plan; no broad sweep.

## Initial verification and limits

- Main was clean and current with `origin/main` at `a4baaa2b8e0d89bd5e69def7d09693ec3d9b034b`.
- Local environment has no GPU tooling, competition data, incumbent predictions, Kaggle credentials or TabPFN token. The owner-supplied screenshot is available; any authentication uses its credential only in process memory, without printing it.
- Private output downloads are limited to needed CSV/JSON evidence first; do not load foreign pickle/joblib/cloudpickle models.
- No experiment job, GPU smoke or submission launched yet. Runtime/configuration/checkpoint identity remains to be pinned after access checks.

## Progress and handoff

Access checks and isolated software verification in progress. Root owns coordination; implementation ownership will be handed to a separate worktree if delegated. Record exact tests, remote versions, artifact hashes, resource use and final decision here and in the shared ledger.
