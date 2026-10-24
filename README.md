# Team and Agent Collaboration Guide

This README is the team's operating guide, not the project report. Every human teammate and AI agent uses the same shared workflow. Read [AGENTS.md](AGENTS.md) before doing work.

**Model training is PAUSED by the owner.** No training, GPU jobs or competition submissions until explicitly resumed. Small synthetic software regression tests are allowed. Project background and results live in [docs/PROJECT.md](docs/PROJECT.md).

## Two non-negotiables

1. **Document your identity and strategy.** Publish your agent name, objective, evidence, approach, branch and exact file ownership before implementation. Keep progress and handoffs current.
2. **Always pull and push.** Pull before starting, read teammates' latest work, commit/push meaningful checkpoints and before stopping, and verify the remote commit. Never push secrets or overwrite teammates' work.

## Where everything goes

| Location | Shared purpose |
|---|---|
| [AGENTS.md](AGENTS.md) | Rules every agent must follow. |
| [docs/STRATEGY.md](docs/STRATEGY.md) | Current direction, agent registry, active ownership and handoffs. Read before choosing work. |
| [docs/plans/](docs/plans/) | One named plan per agent/task: evidence, strategy, scope, steps, verification, status and next action. |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Shared scientific decisions, successful/rejected approaches, provenance and caveats. |
| [docs/ONBOARDING.md](docs/ONBOARDING.md) | Clone/access setup, software tests, artifact limits and housekeeping. |
| [docs/PROJECT.md](docs/PROJECT.md) | Project explanation, code map, verified results, data setup and attribution. |
| Root `*.py` / `test_*.py` | Implementation and regression tests. Preserve this layout unless all consumers are migrated and tested. |
| `data/`, `results/`, `reports/`, `research/`, `kaggle-*/` | Ignored local/private artifacts, NOT synchronized through Git. Publish safe summaries in docs instead. |

## Start a task

1. Use your own clone or isolated worktree, GitHub account and valid Git identity. A public repo provides read access, not write access; see onboarding.
2. Check `git status --short`, run `git fetch origin`, then `git pull --ff-only` on your tracking branch. Inspect current `origin/main`, the registry and active plans. Stop on conflicts rather than discarding changes.
3. Add your own entry in `docs/STRATEGY.md`: name/tool, objective, approach, branch and exact files/areas. Do not change another agent's entry or claim their active files without a handoff.
4. Create `docs/plans/<agent>-<task>.md` using the [template](docs/plans/README.md), link it from your registry entry, then commit/push your claim before implementation.
5. Use `agent/<name>/<task>` for substantial/concurrent code work. Announce/integrate a branch-only claim into the shared main registry before relying on exclusive ownership. A pushed claim is coordination, not an atomic file lock. Keep passing, small maintenance changes on main; keep broken WIP off main.

## While working

- Work in the agreed area and read teammates' changes instead of duplicating them.
- Keep your plan updated with evidence, progress, blockers and changed decisions. Preserve negative findings too.
- Test meaningful changes, commit and push coherent checkpoints. Push unfinished work to your own branch with explicit WIP status.
- Fetch before pushing. Safely integrate relevant upstream changes; never force-push shared branches, reset/stash teammates' work or replace the shared registry wholesale.
- The training pause applies to every agent. A plan or past authorization is not permission to launch another job.

## Finish or hand off

1. Run relevant tests. Record the actual commands, results and limitations in your plan.
2. Update strategy and shared evidence. Mark COMPLETE, PAUSED or BLOCKED, release ownership when done, and name the next action/artifact access needed.
3. Review the staged diff for secrets, restricted data and accidental outputs. Commit, fetch/pull safely, push, and verify the remote branch SHA matches the intended commit.
4. Open a PR for task-branch integration/review. A pushed branch is visible work, not merged work. If push/access is blocked, say so explicitly.

**Work is not handed off if its only plan, result or explanation lives in a private chat.**
