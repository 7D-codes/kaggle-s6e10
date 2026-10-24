# Shared Strategy and Agent Ownership

## Project state

**Training: PAUSED by the owner.** Current phase: collaboration reboot, housekeeping and knowledge transfer. No experiment or competition submission is authorized by this document. Resume only on an explicit owner instruction.

Retained public ROC AUC: **0.96111**, submission **57568517**. Target ≥0.962 is an ambition, not an achieved result or a promise. Read [experiment history](EXPERIMENTS.md) before proposing work.

## Working agreement

Read `AGENTS.md`; pull first, publish your claim, work on nonoverlapping files, push checkpoints and handoff. Keep entries from other agents intact. An unregistered teammate must add their own identity/plan; nobody should invent their entry. Separate clones share Git history, not local ignored files or credentials.

## Agent registry

### Hiro — Hermes / gpt-6.1-sol

- Status: IN PROGRESS, repository reboot. No training jobs.
- Goal: make rules, scientific state and onboarding available to all agents in Git; verify tests do not secretly require this workstation's ignored artifacts.
- Strategy: publish collaboration rules first; audit a source-only checkout; correct only demonstrated onboarding/test defects; publish concise experiment history and artifact-access constraints; remove disposable task caches only.
- Branch: `main` for this bounded maintenance task.
- Owned files: `AGENTS.md`, `README.md`, `.gitignore`, `docs/`, and `test_route_confirmation.py` only if the source-only audit confirms its artifact dependency. Coordinate before touching these while this entry is active.
- Verification: initial pull is current; source-only test audit and final regression pending.
- Handoff: teammate may claim other files/areas after pulling. No scientific/model changes planned.

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
