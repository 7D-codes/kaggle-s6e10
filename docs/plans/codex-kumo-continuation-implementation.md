# Kumo continuation software implementation

Codex / GPT-6 implements only the [frozen continuation contract](codex-kumo-continuation-contract.md) on `agent/codex/next-signal-20261026` in `.worktrees/codex-next-signal`. Root owns launch/scientific decisions. No real-data fit, GPU smoke, downloads, remote jobs, submissions or delegated work here.

## Scope and approach

Own runner resource constants/guard and monotonic receipt timing only, new continuation builder/test, unique strategy subsection and this plan. Preserve old `package_kumo_pair.py` byte-for-byte, all model/feature/context/random/native operations, original tolerances, output/marker guards and CatBoost compatibility. Reuse the existing packager/supervisor with strict replacement checks; build only. Original caps3600/28800 remain default; generated continuation runner/supervisor use5400/24900. Economics remains1.25*5*setup-inclusive elapsed<=24900; prior3606+new24900+reserve294=28800seconds.

Callers inspected with rg: runner consumed by existing Kumo and CatBoost packagers/tests; old Kumo SUPERVISOR also used by CatBoost tests. Risks: accidental numerical change, cap disagreement or weakened process-group cleanup; tests must detect these. No additional dependencies or runtime abstraction planned.

## Progress and verification

Read AGENTS/STRATEGY/EXPERIMENTS/ONBOARDING and current origin/main contract; initial worktree clean. Fetched/pulled tracking branch and merged latest main coordination without dropping task runtime history. Git identity is the existing configured account. Publish ownership before implementation, then targeted RED/GREEN, full suite, fresh source-only archive, staged secret/restricted-file inspection, commit/push and remote SHA readback.

## Handoff

Pending implementation and verification. Native GPU correctness/performance and complete real-data runtime remain outside this synthetic-software task.

### Code checkpoint

Targeted RED: all10new continuation checks failed in2.01seconds for missing builder/default constants/timings (no import errors). Minimal implementation adds default constants3600/28800, substitutes them only in the original guard, and records setup/context and per-arm transform/fit/native-calibration/validation/test/total monotonic durations. Existing v1 packager remains SHA256 `31a15db42a58cfb688d081c02b27cd63c90b25b90bb39d820493a79b0a65c267` byte-identical. New builder reuses its unchanged worker/supervisor, guards unique cap replacements5400/24900, compiles generated source and writes private competition-only T4 metadata. It never launches.

Targeted GREEN:54tests passed (continuation, original Kumo and CatBoost),4.52seconds; subsequent explicit21comparison synthetic assertion and numerical-call AST pin checked at checkpoint. Tests pin every pre-existing non-run function/class AST, computational/native call AST in run, exact generated runner/source bundle/worker/supervisor invariance, unique metadata, economics at3984seconds and just over, all10fresharm constructions, partial failure receipts/markers and blocking-worker process-group kill under both deadlines. Full/source-only verification pending.
