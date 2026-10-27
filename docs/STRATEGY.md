# Shared Strategy and Agent Ownership

## Project state

**New training/evaluation is PAUSED under the owner's latest supplied AGENTS instructions.** Earlier 2026-10-26 resumption authorized the completed CatBoost comparison and existing Kumo launch, but past authorization is not reusable after the newly supplied pause. Read-only research and monitoring of existing kernel141062725/version1 may continue; do not launch, restart or fit a real-data comparison until explicit resumption. Competition submissions remain prohibited. Historical entries retain their original authorization context.

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

### Codex — current direction council

- Status: COMPLETE; advisory only, training remains PAUSED.
- Goal: assess whether the Summit improvement effort is testing incremental signal and identify the next defensible action.
- Evidence/hypothesis and approach: reconcile the completed five-advisor, anonymous-peer-review council with latest owner boundaries; distinguish software/native success, conditional offline gain and actual public improvement.
- Branch: `main` for bounded documentation.
- Ownership released: this registry subsection and `docs/plans/codex-council-current-direction.md`; no runtime files.
- Plan: [current council direction](plans/codex-council-current-direction.md).
- Experiment contract/job owner: none in this task. Historical job status is not current launch permission.
- Verification and handoff: five advisors, five anonymous peer reviews and revised chairman verdict reconciled. Read-only existing CatBoost v1 check at 09:52 UTC: RUNNING, failure message empty; no local terminal outputs yet. Proposed next experiment is bounded canonical-128 Kumo continuation only after explicit resumption, with eight cumulative GPU hours including prior consumption. Documentation links/whitespace/credential scope checked; no runtime tests needed. No job or submission launched.

### Codex — existing CatBoost read-only audit

- Status: read-only audit COMPLETE; score-improvement goal BLOCKED pending explicit training/comparison resumption. Training remains PAUSED.
- Goal: resolve state and artifact completeness of existing CatBoost kernel 140747734/v1 without launching or fitting anything.
- Evidence/hypothesis and approach: last authoritative status RUNNING at 09:52 UTC; poll the same job, preserve terminal outputs, audit source/configuration/native/coverage evidence, report unknowns honestly.
- Branch: `main` for documentation; historical artifacts remain in the bounded-tabpfn worktree.
- Ownership released: this registry subsection and `docs/plans/codex-catboost-readonly-audit.md`; no runtime files.
- Plan: [read-only CatBoost audit](plans/codex-catboost-readonly-audit.md).
- Job identity: existing `wasidyms/s6e10-codex-catboost-ctr-v1/1`; root Codex monitors, no duplicate launch or restart.
- Verification/handoff: API COMPLETE 09:59:34 UTC; 49 own output files preserved. Independent full ID/target/fold/hash/test-average audit passed; all five folds and 60 remote native checks complete, 9965 independently replayed rows exact. Standalone development ROC AUC 0.9610177594499577; ensemble contribution not measured. Quota 27.8253 hours remaining at 10:03 UTC. Cat is integrity-qualified for later control irrespective of its score gate. Frozen comparison fitting awaits explicit resumption clarification; no new fit, job mutation or submission. Documentation and audit self-checks passed; no runtime source change.

### Codex — resumed CatBoost ensemble comparison

- Status: COMPLETE; frozen development comparison passed, no submissions.
- Goal: measure the completed CatBoost member's incremental contribution under the unchanged frozen comparison.
- Evidence/hypothesis and approach: native/full-coverage audit passed; run the existing twenty within-original-fold CPU meta fits once, reproduce unchanged control, enforce predeclared gain/fold/conditional-interval gates without tuning.
- Branch: `main` for documentation; preserved `agent/codex/bounded-tabpfn` for execution/artifacts.
- Ownership released: this subsection, `docs/plans/codex-resumed-catboost-comparison.md` and linked ledger entry. No runtime source edits.
- Plan: [resumed comparison](plans/codex-resumed-catboost-comparison.md).
- Job owner: root Codex; one local four-thread/10800-second CPU comparison. No Cat retraining or GPU launch; no duplicate output/retry.
- Verification/handoff: primary0.96168694246→0.96179190793, gain0.00010496547;5/5positive folds, conditional95%gain interval[0.00007762420,0.00014243317],47.95CPU seconds. Independent full-coverage/AUC/coefficient/candidate-formula audit passed with exact replay. Fixed UNSCORED candidate preserved; no verified Summit/public gain. Next Kumo direction remains bounded by council3.

### Codex — next complementary signal, 2026-10-26

- Status: IN PROGRESS; owner explicitly requests further improvement, no submissions.
- Goal: choose and execute the next bounded complementary experiment after the measured CatBoost gain.
- Evidence/hypothesis and approach: fresh five-perspective/anonymous-review council incorporates the qualified twelve-member control and prior Kumo resource failure; select one falsifiable experiment, publish its fixed contract, verify software/native identity and quota before launch.
- Branch: `main` for decision/contract; isolated task branch for substantial code.
- Owned files/areas: this subsection and `docs/plans/codex-next-signal-20261026.md`; runtime ownership will be handed off explicitly after selection.
- Plan: [next-signal decision and execution](plans/codex-next-signal-20261026.md).
- Job owner/monitor: root Codex only. Private `wasidyms/s6e10-codex-kumo-continuation-v1/1`, kernel141062725, launched23:05:27UTC2026-10-26; RUNNING23:06:05UTC, emptyfailuremessage, remote source/privacy exact. No duplicate/restart; both allowed Astra calls already used.
- Verification/handoff: council4 complete5advisors/5anonymousreviews/chairman, noAstra remaining. [Frozen Kumo continuation](plans/codex-kumo-continuation-contract.md): canonical128/rawFE, qualifiedCatcontrol, gates14vs13and14vs12;24900newworkerseconds/5400firstpair/25%projection/8cumulativeh. Regenerateallpairs,no reuse. Code28d73af SPEC PASS/QUALITY APPROVE,54targeted each; full/freshsource-only179passed. Published branch a6baf34/[draft PR5](https://github.com/7D-codes/kaggle-s6e10/pull/5), packagebe660467…/control91a6fac8… sealed. Prelaunch3d85a6f remoteverified, fresh27.82529h/noactivejobs; one launch confirmed. NativeGPU/firstpair/fullOOF/quality pending; submissions0.
- Post-run comparison preparation: code05b0475 SPEC PASS/QUALITY APPROVE,37targeted/216freshsource-onlypassed; source340fb727…/[draft PR6](https://github.com/7D-codes/kaggle-s6e10/pull/6), no real fits. Retained12control replay/dual gates/failure quarantine frozen. Read-only terminal auditor105e2efe… self-checked. Exact GPUjob stillRUNNING23:24:22UTC, nofailuremessage. Monitor sameversion; no restart, partial promotion or submission.

## New-agent entry template

### Codex — conditional two-device council, 2026-10-27

- Status: COMPLETE; advisory research only, training/evaluation remains paused under the latest owner instructions.
- Goal: decide whether the newly evidenced unused-device possibility warrants a successor proposal after a resource failure, or whether to switch information sources.
- Approach: five independent standard advisors, anonymized peer review and chairman synthesis. Both permitted Astra calls are already consumed. No live-run changes or launches.
- Branch: `main`; ownership released for this subsection, `docs/plans/codex-council-two-device-20261027.md` and linked ledger entry; advisors own no runtime files.
- Plan: [conditional resource decision council](plans/codex-council-two-device-20261027.md).
- Existing job: root monitors141062725/version1 only. New evidence is hardware/source/documentation, not current-session speedup or quality.
- Verification/handoff: five advisors/five anonymous reviews/standard chairman complete, no additional Astra. Chairman deferred successor selection until actual terminal receipts permit a numerical contract; original eight cumulative hours includes BOTH runs and all qualification, with no automatic extension. Peer review added comparative budget value, slower-arm projection, raw-row equivalence and predeclared output eligibility. Exact job remained RUNNING00:03:46UTC. Documentation checks passed; no runtime changes/tests/fits/jobs/submissions.

### Codex — score escalation evidence review, 2026-10-27

- Status: COMPLETE; research and existing-job monitoring only. The owner's newly supplied training-pause instructions supersede historical resumption for new fits/jobs.
- Goal: identify a stronger evidence-backed direction without presenting offline gains as leaderboard improvement.
- Approach: inspect recent public source snapshots, distinguish additional signal from reused blends, and verify the existing Kumo version without changing its frozen experiment.
- Branch: `main`; ownership released for this subsection, `docs/plans/codex-score-escalation-20261027.md` and its linked ledger entry. No runtime ownership.
- Plan: [score escalation evidence review](plans/codex-score-escalation-20261027.md).
- Existing job: root monitors `wasidyms/s6e10-codex-kumo-continuation-v1/1`, kernel141062725; RUNNING with empty failure message at23:40:35UTC2026-10-26. No launch/restart in this task.
- Verification/handoff: four API notebook-source hashes checked; latest public work supplies no newly qualified complementary member. Published source-specific limits and next-evidence decision; documentation checks passed, runtime tests unnecessary. Existing Kumo result still pending; no real-data fit, GPU launch or submission. Monitor exact version and obtain resumed evaluation scope before any further fitting.
- Resource follow-up: earlier hardware receipt proves two T4s, current Kumo code selects GPU0. Official specifications and a separate project's measurement support investigating independent work on both cards, but speedup/current-account quota charging/native equivalence remain unmeasured. Research-only lead for a later council after a resource failure; no change to the live experiment and no successor selected.

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

- The owner previously resumed Codex training/evaluation on 2026-10-26. The latest supplied AGENTS instructions pause new fitting/jobs again; retain historical results and monitor the existing version without restarting. Submission permission was not granted.
- Current diagnosis suggests limited model diversity; TabPFN is only an unverified research lead. Its existence is not permission to install/download/launch it.
- Preserve historical negative results and the incumbent. No new agent should start a fresh feature/epoch sweep by default.


### Codex Airline Handoff — Codex / GPT-6

- Status: COMPLETE; findings and saved-model transfer published. Training remains PAUSED; no training/scoring/job/Kaggle submission in this handoff task.
- Goal/result: separate owner-authorized Windows campaign handoff; eligible observed public ROC AUC **0.96119**, with comparison 0.96181/goal 0.96213 unmet. Retained Batch 6 evidence elsewhere remains unchanged.
- Approach/evidence: preserve exact recipe, positive and negative trials, source/model hashes, replay/composition audits and adaptive-validation/portability caveats. Keep binaries and raw predictions outside Git history.
- Branch: `agent/codex/airline-handoff-20261025`; [PR #4](https://github.com/7D-codes/kaggle-s6e10/pull/4) is the integration/review path; not claimed merged.
- Plan: [completed handoff plan](https://github.com/7D-codes/kaggle-s6e10/blob/3a47b819c55d2a4410bb5401a4b376fdd376f70d/docs/plans/codex-airline-handoff-20261025.md).
- Findings: [separate report and manifests](https://github.com/7D-codes/kaggle-s6e10/tree/3a47b819c55d2a4410bb5401a4b376fdd376f70d/docs/handoffs/airline-self-trained-20261025).
- Models: [published release](https://github.com/7D-codes/kaggle-s6e10/releases/tag/airline-self-trained-20261025); 103 model/encoder/context artifacts in 17 ZIPs, plus own submitted CSV and hash manifests. All 20 GitHub asset digests and byte sizes match local files.
- Verification: original provenance hashes, 17 ZIP CRCs, 444 archive-entry SHA256 checks, document/relative-link checks and remote branch readback passed. Runtime code unchanged. Exact source modules and external licensed inputs remain prerequisites; fresh-platform portability not fully tested.
- Ownership released: own registry subsection, handoff directory, plan and appended ledger entry. Other agents' entries and historical scientific records preserved.
- Handoff: review/integrate PR #4. All models/code/evidence preserved. No extra training authorization is implied by this transfer.
