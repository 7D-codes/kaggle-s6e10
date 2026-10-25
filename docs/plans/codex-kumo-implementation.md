# Frozen paired Kumo implementation

Root contract: [paired Kumo](codex-kumo-paired.md). Owned runtime files: train_kumo_pair.py, package_kumo_pair.py, test_kumo_pair.py. No real-data fit/model installation/download/job/submission in this task.

## Scope and approach

Use native Kumo small classification with fixed two-estimator cached inference and universal fixed128 dispatch. Pure feature state stores context categories/route statistics/cross-fit TE; native schema remains frozen to four categorical columns. Both arms share selected context rows. Package embeds existing dispatcher source and a process-group supervisor with setup-inclusive first-pair and total deadlines. Fail closed for hashes, identity, incomplete coverage and native replay drift.

Callers: new runner CLI and private package only. Existing source-only tests must stay independent of torch/SDM. New tests cover feature isolation, categories, TE exclusion, paired selection, replay, deadlines, hash/ID/coverage gates and package compilation.

## Progress and verification

- Read project coordination/onboarding and native source pinned by root. Git fetch requires escalated metadata writes and succeeded; initial task branch has no upstream.
- Implementation claim precedes code; synthetic TDD and final source-only suite pending.
- Native runtime/checkpoint loading and GPU feasibility are deliberately unexecuted; independent root reviews and launch ownership remain required.
