# Shared Strategy and Agent Ownership

## Project state

**Training: PAUSED by the owner.** Current phase: collaboration reboot, housekeeping and knowledge transfer. No experiment or competition submission is authorized by this document. Resume only on an explicit owner instruction.

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

- Training remains paused while repository collaboration and shared evidence are established.
- Current diagnosis suggests limited model diversity; TabPFN is only an unverified research lead. Its existence is not permission to install/download/launch it.
- Preserve historical negative results and the incumbent. No new agent should start a fresh feature/epoch sweep by default.


### Codex Airline Handoff — Codex / GPT-6

- Status: IN PROGRESS; owner identified the signed-in Chrome profile with repository write access. Publishing this bounded handoff claim through GitHub UI; artifact export and verification are complete.
- Goal: add the owner's completed Windows self-trained campaign findings and saved models as a separate handoff, without modifying the retained Batch 6 evidence or restarting training.
- Evidence/approach: preserved receipt records a separate eligible public ROC AUC 0.96119; retain failed reference trials, paired-validation caveats, actual code/model hashes and exact recipe. Store model archives as authorized release assets, with public-safe findings/manifests in docs.
- Branch: `agent/codex/airline-handoff-20261025`.
- Owned files/areas requested: this registry subsection, `docs/plans/codex-airline-handoff-20261025.md`, `docs/handoffs/airline-self-trained-20261025/`, and the appended separate-campaign link in `docs/EXPERIMENTS.md`. Other agents' entries and runtime modules remain untouched.
- Plan: [codex-airline-handoff-20261025.md](plans/codex-airline-handoff-20261025.md).
- Experiment/job owner: none. No training, GPU jobs, scoring or Kaggle submission in this handoff task.
- Verification: 103 existing model/encoder/context artifacts grouped into 17 ZIPs; original submitted-provenance hashes and ZIP CRC checks passed. No competition CSV data or vendor checkpoint copied. An own submitted-prediction CSV is prepared only as a release asset. Git fetch/pull succeeded; noninteractive push dry-run failed because no credentials are available.
- Handoff: publish this registry claim, then the task-branch handoff and PR; upload release assets and verify remote artifacts. Git CLI credentials are unavailable, so use the owner's existing authenticated GitHub browser session.
