# Shared Agent Rules

Applies to every AI agent and subagent working in this repository. Read this file, `docs/STRATEGY.md`, and `docs/EXPERIMENTS.md` before starting. User instructions override these project defaults.

## Current boundary: training is paused

The owner has paused model training for a collaboration/project reboot. Do not launch training, GPU smokes, remote jobs, hyperparameter searches, or competition submissions until the owner explicitly resumes that scope. Tiny synthetic unit-test fits are allowed for software verification, not score improvement. Research and repository maintenance may continue. Past job/submission permission is not reusable authorization.

## 1. Pull first, push your work

- At the beginning of a session and before claiming work: inspect `git status --short`, `git fetch origin`, and `git pull --ff-only` on your tracking branch. Read the latest `origin/main` coordination documents too. Do not overwrite or stash somebody else's uncommitted work.
- Publish your strategy/ownership claim before implementation so teammates can see it. Use small coherent commits and push after each meaningful checkpoint and before stopping or handing off. Work is not shared until its remote branch contains it.
- Fetch again before pushing. On `main`, pull with `--ff-only`; if the branch diverges, stop and reconcile deliberately. On a task branch, inspect new `origin/main` changes and integrate relevant changes before declaring completion. Never blindly overwrite coordination files.
- Keep passing maintenance/docs changes on `main` small. Use a unique branch such as `agent/<name>/<task>` for substantial code changes, uncertain work, or concurrent overlapping work. Push unfinished work to that branch with a clear WIP status, not broken code to `main`; open a PR for integration/review.
- Verify the remote branch SHA equals the intended local commit after every push. If blocked by auth, conflicts or network, record the exact unpublished state and report it. Never say “pushed” without readback.
- Never force-push shared branches, reset/discard teammates' changes, rewrite shared history, or use `git clean` as housekeeping. “Always push” never permits publishing credentials, restricted data, or unsafe changes.

## 2. Identify yourself and document the strategy

In `docs/STRATEGY.md`, add/update your own entry before work:

- Agent name and tool/model when known; never invent another agent's identity.
- Objective, evidence/hypothesis and planned approach.
- Branch and exact owned files/areas, status, verification and next handoff.
- Job identity if authorized later: kernel/version, launch owner and monitor. No duplicate launches.

Use your real configured Git account identity; record the AI name in the strategy entry, not as a fabricated human commit author. Add an `Agent: <name>` commit-message trailer where practical. Update your entry at checkpoints and closeout, including negative results and blockers. The Git commit timestamps provide the chronology.

No two agents may modify the same owned files without an explicit handoff. If a published claim overlaps, coordinate first. Prefer separate clones, or worktrees nested under `.worktrees/`; never have independent agents write into the same working directory. GitHub is the shared coordination/source environment, not a shared live filesystem. A push race is resolved by pulling and merging distinct entries, never by last-writer-wins replacement. Ownership entries are advisory coordination, not atomic locks.

## 3. Shared evidence, not private-session memory

Keep team-readable decisions, experiment contracts, results and rejected approaches in `docs/EXPERIMENTS.md` or linked small files under `docs/`. A local ignored report or chat message alone is not a team handoff. Include configurations, split seed, runtime versions, metric definition, decision and artifact provenance; mark unknowns explicitly.

Do not commit competition datasets, raw predictions, checkpoints, credentials, private logs, or copied third-party notebooks. `data/`, `results/`, `reports/`, `research/` and generated `kaggle-*/` are intentionally local-only. Share public-safe summaries/hashes and authorized remote artifact locations instead. Private Kaggle outputs require owner-granted access; do not bypass it. Never load untrusted pickle/joblib/cloudpickle files. Read `docs/ONBOARDING.md` for access and setup limitations.

## 4. Scientific and code standards

- The metric is ROC AUC, not accuracy. Separate development OOF, robustness repeats, untouched validation, and actual public scores. Never promise a score or infer rank from OOF.
- Current retained public result: **0.96111**, submission **57568517**. Preserve its evidence and predictions; do not promote an experiment solely because its development point estimate is higher.
- After training is explicitly resumed, predeclare hypothesis, unchanged control, fold seed, parameters, fixed blend, resource cap, promotion criteria and stop rule before launching. Do not repeat rejected work without new evidence. Submission still requires its own explicit authorization.
- Fit statistics/categories only within the training fold; cross-fit target encoders. Exclude original-data overlaps with both competition train/test. Never pass outer validation labels to fixed-epoch neural fitting. Never blend mismatched OOF folds or silently tune weights on exposed rows.
- Use TDD for fixes: reproduce the failure, make the smallest readable correction, then run targeted and full tests. Test a fresh source-only checkout, not only a workstation with private artifacts. Native inference tests must be nonconstant and preserve the original declared tolerance.
- Preserve package/source/configuration hashes and raw-row replay caveats. A hash match does not prove equivalent inference. Do not relax a failing tolerance to manufacture acceptance.
- Reuse current modules and standard tools. Developer-agent tooling is not a runtime dependency. Avoid broad refactors or moving modules unless imports, packagers and tests are migrated together.

## 5. Housekeeping and closeout

Remove only regenerable, task-owned caches/fixtures that no active process uses. Preserve datasets, trained models, native preprocessing, predictions, source snapshots, manifests, audit evidence and unfinished work. Do not delete useful local artifacts merely because Git ignores them; do not perform machine-wide cache/Docker cleanup under a repo-maintenance request.

Before handoff: run the relevant tests, update your strategy and evidence, inspect the staged diff for secrets/restricted files, commit and push, and verify the remote SHA. State exactly what changed, what was tested, what remains blocked, and whether any job/submission occurred.
