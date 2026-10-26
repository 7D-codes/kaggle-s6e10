# Kumo continuation software implementation

Codex / GPT-6 implements only the [frozen continuation contract](codex-kumo-continuation-contract.md) on `agent/codex/next-signal-20261026` in `.worktrees/codex-next-signal`. Root owns launch/scientific decisions. No real-data fit, GPU smoke, downloads, remote jobs, submissions or delegated work here.

## Scope and approach

Own runner resource constants/guard and monotonic receipt timing only, new continuation builder/test, unique strategy subsection and this plan. Preserve old `package_kumo_pair.py` byte-for-byte, all model/feature/context/random/native operations, original tolerances, output/marker guards and CatBoost compatibility. Reuse the existing packager/supervisor with strict replacement checks; build only. Original caps3600/28800 remain default; generated continuation runner/supervisor use5400/24900. Economics remains1.25*5*setup-inclusive elapsed<=24900; prior3606+new24900+reserve294=28800seconds.

Callers inspected with rg: runner consumed by existing Kumo and CatBoost packagers/tests; old Kumo SUPERVISOR also used by CatBoost tests. Risks: accidental numerical change, cap disagreement or weakened process-group cleanup; tests must detect these. No additional dependencies or runtime abstraction planned.

## Progress and verification

Read AGENTS/STRATEGY/EXPERIMENTS/ONBOARDING and current origin/main contract; initial worktree clean. Fetched/pulled tracking branch and merged latest main coordination without dropping task runtime history. Git identity is the existing configured account. Publish ownership before implementation, then targeted RED/GREEN, full suite, fresh source-only archive, staged secret/restricted-file inspection, commit/push and remote SHA readback.

## Handoff

Pending implementation and verification. Native GPU correctness/performance and complete real-data runtime remain outside this synthetic-software task.
