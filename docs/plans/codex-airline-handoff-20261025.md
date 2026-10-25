# Completed self-trained Windows campaign handoff

- Agent/tool/model: Codex / GPT-6.
- Status: COMPLETE. Handoff documents and models published; final repository review/integration is tracked in PR #4. No training or submission restarted.
- Branch: `agent/codex/airline-handoff-20261025`.
- Goal and strategy: preserve the owner's completed campaign as a separate, reproducible evidence handoff. Add findings/recipe/manifests under `docs/handoffs/airline-self-trained-20261025/`; keep binary checkpoints and raw submitted predictions out of Git history and attach them to an authorized GitHub Release.
- Evidence: saved Complete receipt verifies public ROC AUC 0.96119; source/model/encoder and replay/composition provenance are preserved. Exact0.96213 copied ensemble was not reproduced. Both comparison 0.96181 and requested 0.96213 goals remain unmet.
- Owned areas: own registry entry, this plan, the handoff directory and a small appended separate-campaign link in `docs/EXPERIMENTS.md`.
- Boundaries: no training, prediction generation, quality rescoring, Kaggle submissions, changes to other agents' historical findings, competition datasets, vendor checkpoint redistribution, credentials or browser/private logs.

## Progress

- [x] Clone repository; read AGENTS, strategy, experiments and onboarding; fetch and fast-forward pull.
- [x] Inspect completed campaign artifacts and preserve hashes.
- [x] Build 17 ZIPs containing103 model/encoder/context artifacts, plus source/evidence; verify ZIP CRCs and original evidence hashes.
- [x] Prepare separate report, exact five-recipe blend, per-file model hashes, source hashes and restore/version caveats.
- [x] Prepare the actual own submitted CSV as an external release asset only.
- [x] Published the bounded strategy/claim to main through the signed-in owner account; remote commit `a170ee859932a6c12331de38320e4f65774c5671` read back with Git fetch/show.
- [x] Published through the owner's existing authenticated browser session; verified remote branch and document contents. Opened [PR #4](https://github.com/7D-codes/kaggle-s6e10/pull/4). Eight uploaded documents are byte-exact; README text matches after normal LF normalization by the GitHub editor.
- [x] [Published model release](https://github.com/7D-codes/kaggle-s6e10/releases/tag/airline-self-trained-20261025). GitHub server digests and exact sizes match all 20 planned assets; publication receipt saved in the handoff directory.

## Verification

- `git fetch origin` succeeded; `git pull --ff-only` reported already up to date.
- Export command: `.venv/Scripts/python.exe exports/build_7d_handoff.py` in the preserved Windows campaign workspace. Passed original provenance SHA-256 validation, credential-literal scan, all ZIP CRCs and per-archive size guard (<2 GiB).
- Archives total 2932337674 bytes before the separately prepared submitted CSV. See `package_verification.json` for exact asset hashes/sizes.
- `git -c credential.interactive=false push --dry-run ...` failed with no password available and interactive prompts disabled. The ownership claim was subsequently published through the authenticated browser; all 20 release assets are now published and verified.
- A second independent archive-content verification passed SHA-256 checks on all 444 entries (103 model/encoder/context files plus 341 source/evidence entries), plus every archive SHA and the own submitted CSV hash. Relative Markdown links passed. No binary model or raw CSV is in the Git-tracked handoff directory.
- `git diff --cached --check` passed.
- Runtime tests were not run: only a documentation/artifact handoff is being prepared; repository runtime code is unchanged.

## Decisions and handoff

The main repository's retained 0.96111/57568517 evidence is a separate historical campaign and remains unchanged. This new handoff documents the owner's distinct Windows campaign and its observed 0.96119 result with limitations. Preserve negative experiments and label public/DEV metrics separately. Models are trusted campaign artifacts; no serialized objects were loaded or retrained during export.

Local prepared artifacts are indexed by the handoff manifests. The published release tag is `airline-self-trained-20261025`. Server SHA-256/size verification passed for all 20 assets. No runtime code changed; ownership is released after the handoff. PR #4 is the integration/review path; it is not claimed merged. CLI authentication was not changed or created; all writes used the owner's existing browser session.

- Final transfer command: `.venv/Scripts/python.exe exports/verify_7d_release.py` — passed all 20 server SHA-256/size comparisons.
