# Kumo continuation software implementation

Codex / GPT-6 implements only the [frozen continuation contract](codex-kumo-continuation-contract.md) on `agent/codex/next-signal-20261026` in `.worktrees/codex-next-signal`. Root owns launch/scientific decisions. No real-data fit, GPU smoke, downloads, remote jobs, submissions or delegated work here.

## Scope and approach

Own runner resource constants/guard and monotonic receipt timing only, new continuation builder/test, unique strategy subsection and this plan. Preserve old `package_kumo_pair.py` byte-for-byte, all model/feature/context/random/native operations, original tolerances, output/marker guards and CatBoost compatibility. Reuse the existing packager/supervisor with strict replacement checks; build only. Original caps3600/28800 remain default; generated continuation runner/supervisor use5400/24900. Economics remains1.25*5*setup-inclusive elapsed<=24900; prior3606+new24900+reserve294=28800seconds.

Callers inspected with rg: runner consumed by existing Kumo and CatBoost packagers/tests; old Kumo SUPERVISOR also used by CatBoost tests. Risks: accidental numerical change, cap disagreement or weakened process-group cleanup; tests must detect these. No additional dependencies or runtime abstraction planned.

## Progress and verification

Read AGENTS/STRATEGY/EXPERIMENTS/ONBOARDING and current origin/main contract; initial worktree clean. Fetched/pulled tracking branch and merged latest main coordination without dropping task runtime history. Git identity is the existing configured account. Publish ownership before implementation, then targeted RED/GREEN, full suite, fresh source-only archive, staged secret/restricted-file inspection, commit/push and remote SHA readback.

## Handoff

Implementation completed and runtime ownership handed back to root for independent SPEC then QUALITY review. Native GPU correctness/performance and complete real-data runtime remain outside this synthetic-software task.

### Code checkpoint

Targeted RED: all10new continuation checks failed in2.01seconds for missing builder/default constants/timings (no import errors). Minimal implementation adds default constants3600/28800, substitutes them only in the original guard, and records setup/context and per-arm transform/fit/native-calibration/validation/test/total monotonic durations. Existing v1 packager remains SHA256 `31a15db42a58cfb688d081c02b27cd63c90b25b90bb39d820493a79b0a65c267` byte-identical. New builder reuses its unchanged worker/supervisor, guards unique cap replacements5400/24900, compiles generated source and writes private competition-only T4 metadata. It never launches.

Targeted GREEN:54tests passed (continuation, original Kumo and CatBoost),4.52seconds; subsequent explicit21comparison synthetic assertion and numerical-call AST pin checked at checkpoint. Tests pin every pre-existing non-run function/class AST, computational/native call AST in run, exact generated runner/source bundle/worker/supervisor invariance, unique metadata, economics at3984seconds and just over, all10fresharm constructions, partial failure receipts/markers and blocking-worker process-group kill under both deadlines. Full/source-only verification pending.


### Final source-only verification and handoff

Code checkpoint `28d73afd09d85ed96588f0020fbd071aa5aed529` was pushed and exact remote branch SHA read back. Final targeted54tests passed4.49seconds, including21nonconstant synthetic native comparisons per arm at original tolerances. Full source checkout179passed/66existingwarnings35.93seconds; independent fresh `git archive` source-only tree `/tmp/codex-kumo-continuation-source-28d73af`179passed/66existingwarnings35.85seconds. No data/results/models/private manifests were copied into the archive. Used existing test environment: Python3.13.11, NumPy2.5.4, pandas3.1.0, sklearn1.9.1, SciPy1.18.1, CatBoost1.2.8, pytest9.1.1. These software-test versions do not change pinned remote runtime versions.

Generated/compiled package path `.worktrees/codex-next-signal/kaggle-kumo-continuation/train.py` (local ignored artifact, no launch). Public-safe SHA256 provenance:

- Generated train.py: `be660467fadff101cd49655f1258ab3c794dc329426e24103971ae5d4a756c3f`.
- Generated metadata: `380b0cde512aacc13319a688b799adf6c3f0790ae452d0fd03db348bf3cac238`.
- Default runner: `57666fdf19d24a03bd0e25243e3688a9c00918f9d145bcae504ff50af7d25de4`.
- Unchanged dispatcher: `6880f4798d6f44d50386e763c424f0824cd2dc9ddb6eeb95e341ef6affee618a`.
- New builder: `393850992e93b773706b42d9874d6b84e22e013d3c1c71566d5ffc82f9de2e6c`.

Fetched again before closeout; no new origin/main commits remained to integrate. Staged file list/diff/whitespace inspected: only the five handed-off files, no datasets, predictions, credentials or restricted artifacts. No real-data fits, native GPU smokes, source/weight downloads, Kaggle jobs/submissions or delegated work occurred. No score improvement claimed. Root retains all launch/monitor/scientific/export decisions; resource feasibility and native T4 behavior remain pending real execution under the frozen cap. Stage timings include model construction in fit/cache and native archive in native calibration; setup includes worker setup plus initial input preparation, while context preprocessing is recorded separately per fold. Failed Python stages keep their elapsed durations and prior receipts; hard-killed native stages retain the last saved completed-stage timings as before.
