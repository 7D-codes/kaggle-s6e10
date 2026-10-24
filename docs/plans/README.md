# Agent Task Plans

Create `<agent>-<task>.md`, for example `hiro-repository-reboot.md`. Link it from `../STRATEGY.md`, publish before implementation and update through handoff. No private logs, datasets or model files here.

## Template

```markdown
# <Task>
- Agent/tool/model:
- Status: PLANNED / IN PROGRESS / BLOCKED / PAUSED / COMPLETE
- Branch:
- Goal and strategy:
- Evidence / why this approach:
- Owned files/areas:
- Boundaries and non-goals:

## Progress
- [ ] ...

## Verification
Actual commands/results and limitations. Distinguish planned checks from completed checks.

## Decisions and handoff
Results, rejected options, next action, ownership release, artifact paths/access.
```

A training plan remains a proposal during the owner's pause. It does not authorize a job. Preserve teammates' entries and reconcile collisions rather than overwriting.
