# Teammate and Agent Onboarding

## Start here

1. Clone `https://github.com/7D-codes/kaggle-s6e10.git` into your own directory. Read [AGENTS.md](../AGENTS.md), [shared strategy](STRATEGY.md), and [experiment ledger](EXPERIMENTS.md).
2. Confirm GitHub write access and your own valid Git identity with `gh auth status`, `git config user.name`, and `git config user.email`. A public repository grants read access, not push access. The owner must invite your GitHub account if direct collaboration is intended. Never reuse another person's token or identity.
3. Pull, register your agent name, strategy, branch and owned files in the shared strategy, commit and push your claim. If you cannot push, say so before relying on a claim being shared. A fork/PR is an alternative; its unmerged claim is not an accepted exclusive reservation.
4. Keep training paused. Start with research, documentation or software maintenance in your claimed area. Synthetic regression-test fits are allowed; do not launch real experiment/GPU jobs.

## Environment and tests

Use Python3.11+ and uv. The current smoke/regression suite is intended to work from source alone with synthetic fixtures, without Kaggle credentials, datasets, private model bundles, CUDA or PyTabKit training installs:

```bash
uv run --with pandas --with scikit-learn --with pytest --with joblib \
  --with lightgbm --with xgboost --with cloudpickle==3.1.2 \
  python -m pytest -q
```

On macOS, install libomp if the LightGBM import requires it. Reuse the same dependency set rather than repeatedly creating many heavyweight isolated environments. This command tests software; it does not reproduce historical training/runtime pins. For future training, use each experiment's recorded versions and explicitly authorized resource budget.

## Data and artifact access

- Follow [project data setup](PROJECT.md#data-setup) for official competition/original downloads. Accept competition terms and authenticate using your own account; do not redistribute competition CSVs through Git.
- Fresh clones do not contain `data/`, `results/`, `reports/`, `research/` or generated `kaggle-*/` outputs. These are intentionally ignored, not missing Git commits.
- Shared public-safe facts and provenance belong in `docs/`, not solely in ignored reports. Full artifacts remain in the project owner's local archive and/or the linked private Kaggle kernels. Ask the owner for appropriate Kaggle access or an authorized artifact transfer when needed; no shared artifact bucket is configured.
- The linked private kernel versions are evidence/artifact locations, not publicly accessible deliverables. Do not download large checkpoint bundles before checking both permission and storage headroom.
- `package_route_confirmation.py` and `package_route_replay.py` require the preserved Batch6 inference manifest under `results/`. Other historical packagers can require fixed prediction/teacher/source artifacts at packaging or remote execution time. They are historical reproduction tools, not generic fresh-clone commands. Do not fabricate replacement scientific manifests or run them to start new jobs while training is paused.
- Never load untrusted pickle/joblib/cloudpickle files. Verify hashes and trusted provenance before artifact replay. Preserve the CPU/CUDA caveat in the experiment ledger.

## Layout and housekeeping

Keep current Python modules at the repository root for now: imports, tests and embedded Kaggle packagers depend on this layout. Do not reorganize files for appearance without migrating and verifying all consumers. `docs/` is the team-visible knowledge layer. Generated outputs remain ignored. `.worktrees/` is the optional nested location for isolated worktrees, not a required extra environment.

Safe removals are task-owned `__pycache__`, `.pytest_cache` and completed synthetic temporary test directories after checking no process uses them. Data, checkpoints, native states, predictions, manifests, source snapshots and negative-result evidence are not disposable. No machine-wide dependency/cache/Docker cleanup is authorized by repo housekeeping.

## End a work session

Update your strategy entry and shared evidence, run the relevant tests, review the staged diff for secrets, commit, fetch/pull safely, push, verify the remote SHA and state any remaining blocker. Keep WIP on your own branch. Do not leave the only explanation of your work in a private chat.
