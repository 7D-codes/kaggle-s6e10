# Shared Strategy and Agent Ownership

## Project state

**Training: RESUMED for Codex's bounded breakthrough plan by the owner's explicit instruction, “great go ahead and do this plan.”** Scope: artifact recovery, a capped TabPFN pilot and gated evaluation, then the conditional TabM/combination stages in the linked plan. Competition submissions remain unauthorized. This is not permission for unrestricted searches or duplicate jobs.

Retained public ROC AUC: **0.96111**, submission **57568517**. Target ≥0.962 is an ambition, not an achieved result or a promise. Read [experiment history](EXPERIMENTS.md) before proposing work.

## Working agreement

Read `AGENTS.md`; pull first, publish your claim, work on nonoverlapping files, push checkpoints and handoff. Keep entries from other agents intact. An unregistered teammate must add their own identity/plan; nobody should invent their entry. Separate clones share Git history, not local ignored files or credentials.

## Agent registry

### Hiro — Hermes / gpt-6.1-sol

- Status: BLOCKED, optional protected-rule clarification awaits user approval. Training remains PAUSED; no jobs/submissions launched.
- Review follow-up: reviewer inspected the earlier snapshot where Hiro owned all of `docs/`. Current ownership was already released before the review arrived, so that onboarding conflict is not present in current main. An attempted explicit self-registration exception for future broad claims was blocked by the protected `AGENTS.md` approval gate (timeout); no rule edit was applied. Do not retry or bypass that gate without user approval.
- Plan: [hiro-repository-reboot.md](plans/hiro-repository-reboot.md).
- Goal: make rules, scientific state and onboarding available to all agents in Git; verify tests do not secretly require this workstation's ignored artifacts.
- Strategy: publish collaboration rules first; audit a source-only checkout; correct only demonstrated onboarding/test defects; publish concise experiment history and artifact-access constraints; remove disposable task caches only.
- Branch: `main` for this bounded maintenance task.
- Ownership released again: no files remain reserved by Hiro. Historical changes: `AGENTS.md`, `README.md`, `docs/` and the demonstrated test artifact dependency in `test_route_confirmation.py`.
- Verification: source-only RED reproduced the private-manifest dependency; corrected full source-only suite passed48tests (66existing warnings,13.70s). Post-cleanup smoke regression passed1test. Local documentation links checked. No private data/models required for the suite.
- Handoff: pull current main, read the collaboration-only README, register your own name/strategy and create a plan before work. Teammate GitHub username is still needed to resolve the requested collaborator invite. Scientific artifacts preserved; broad disk recovery remains out of scope.

### Codex — GPT-6

- Status: COMPLETE; research only, training remains PAUSED.
- Goal: identify evidence-backed routes from retained public ROC AUC 0.96111 to above 0.96200.
- Evidence/hypothesis and approach: review accepted/rejected experiments, inspect current primary competition/model sources, distinguish independent model signal from leaderboard-selected blends, and rank bounded future experiments.
- Branch: `main` for bounded documentation research.
- Ownership released: this registry entry, `docs/plans/codex-breakthrough-research.md`, and the linked research-only ledger entry in `docs/EXPERIMENTS.md`; no runtime files changed.
- Plan: [codex-breakthrough-research.md](plans/codex-breakthrough-research.md).
- Experiment contract/job owner: none; no training or submission authorized.
- Verification and result: reviewed six public notebook sources with version/hash provenance and official model documentation; matched historical TabPFN source hash and identified API version 2. Published a ranked queue, fold-safety findings, proposed compute caps and promotion gates. Documentation links, whitespace and credential patterns checked; no runtime tests needed or run.
- Next handoff/blockers: recover exact incumbent OOF artifacts and finalize a bounded TabPFN contract after explicit training resumption. No measured improvement yet; private reports/results, available GPU quota and hardware feasibility remain unverified. Supplied credential was not used or saved. No job/submission occurred.

### Codex execution — Codex / gpt-6.1-sol

- Status: IN PROGRESS; bounded training authorized by the latest owner instruction; submissions unauthorized.
- Goal: execute the sourced breakthrough plan, starting with exact comparator recovery and a TabPFN feasibility gate.
- Evidence/hypothesis: in-context predictions may add signal to the retained three-model ensemble; no candidate gain is established.
- Branch: `agent/codex/bounded-tabpfn`; isolated worktree `.worktrees/codex-tabpfn`.
- Ownership: this entry/project-state update, `docs/plans/codex-bounded-tabpfn.md`, linked ledger evidence and task-specific experiment runner/packager/tests, with explicit handoff before any subagent edits.
- Plan: [codex-bounded-tabpfn.md](plans/codex-bounded-tabpfn.md).
- Job owner/monitor: Codex; none launched. Prelaunch contract will pin model/package/source identity, folds, fixed blend and resource cap.
- Verification: clean main, fetched/pulled latest remote documents; no local GPU, datasets, incumbent outputs or installed Kaggle credentials. Isolated dependency environment being prepared.
- Next: authenticate using the owner-supplied token without logging it, inspect private output metadata, recover only needed safe-format artifacts and establish model access/hardware feasibility. Publish checkpoint before launch.

### Codex TabPFN runner — Codex / GPT-6

- Status: IN PROGRESS fixed FP32 correction arm by explicit root handoff after reported native autocast batch-size replay failure; tolerances unchanged. Root retains sole launch/monitor ownership.
- Goal: implement a bounded fold-local raw-feature feasibility pilot and private Kaggle packager.
- Evidence/hypothesis and approach: synthetic TDD for first-fold/context selection, categorical isolation, strict outputs and supervised total deadline; no score claim.
- Branch: `agent/codex/tabpfn-runner`; isolated worktree `.worktrees/codex-tabpfn-code`.
- Narrow ownership reclaimed by root handoff for FP32 correction only: new `train_tabpfn.py`, `package_tabpfn_pilot.py`, `test_tabpfn.py`, `test_tabpfn_pilot.py`, this registry entry and [runner plan](plans/codex-tabpfn-runner.md), by explicit root handoff.
- Experiment contract/job owner: none here; no downloads, training, GPU jobs or submissions by this agent.
- Verification: read coordination, execution plan and onboarding; clean task worktree; fetched origin and fast-forwarded from parent task branch (new branch initially had no upstream).
- Verification: baseline 48 tests passed; 22 new synthetic tests passed. Final fresh source-only archive: 70 tests passed, 66 existing warnings, 13.66s at code checkpoint `82b586c`. No model/data downloads, real training, GPU job or submission.
- Next: root specification/quality review and integration; native TabPFN/CUDA feasibility and replay remain unverified until the separately owned bounded pilot. No full OOF or score-improvement claim.

## New-agent entry template

```markdown
### <agent name> — <tool/model, if known>
- Status: PLANNED / IN PROGRESS / BLOCKED / COMPLETE / PAUSED
- Goal:
- Evidence/hypothesis and approach:
- Branch:
- Owned files/areas:
- Experiment contract/job owner: none, unless explicitly authorized
- Verification and result:
- Next handoff/blockers:
```

## Decisions

- The prior general pause is superseded only for Codex's owner-authorized bounded execution; other unclaimed training/search/submission work is not authorized.
- Current diagnosis suggests limited model diversity; TabPFN is only an unverified research lead. Its existence is not permission to install/download/launch it.
- Preserve historical negative results and the incumbent. No new agent should start a fresh feature/epoch sweep by default.
