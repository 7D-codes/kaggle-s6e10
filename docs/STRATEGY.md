# Shared Strategy and Agent Ownership

## Project state

**Training/evaluation RESUMED for Codex's score-improvement effort on 2026-10-26.** The owner explicitly said "GO AHEAD YOU HAVE FULL RIGHTS" after the pause discussion. Start with the frozen CatBoost ensemble comparison; any later GPU experiment retains its preregistered checks/caps and sole launch owner. Competition submissions remain prohibited. Historical PAUSED entries below describe their original tasks, not the new authorization.

**Latest active-goal authorization: training RESUMED for root Codex's score-improvement work using available Kaggle quota; submissions prohibited.** The owner requires llm-council at major decisions and allows at most two Astra calls. [Current owned goal/evidence](plans/codex-summit-score-goal.md). Historical pre-pilot quota was29.88344hours; current preflight2026-10-25T21:31UTC reports27.82529hours remaining and no active jobs. Original objective remains unfulfilled; an unscored file is not a gain.

Direct Summit Composer adaptation COMPLETE:11public inputs hash-verified, original CSV/ranking reproduced exactly, one unscored derivative changes TabPFNshare8%to7%. Edited notebook and candidate exported locally, with exact fresh-process replay. No training, GPU job or submission. [Evidence/artifacts](plans/codex-summit-composer-adaptation.md).

Historical own-model public ROC AUC: **0.96111**, submission **57568517**, retained for reproducible control/evidence. The owner's supplied public-code result already exceeds0.96200 and is the primary benchmark below. Read [experiment history](EXPERIMENTS.md) before proposing work.

**Primary working benchmark: the owner's supplied Summit Composer public-code result, already above0.96200.** Its pinned version describes0.96213public. Preserve0.96111 as historical own-model evidence; future candidate improvement must target Summit. See [corrected baseline/approach](plans/codex-summit-improvement.md). Exact owner submission receipt remains unknown; no independent rescore is claimed.

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

- Status: IN PROGRESS on the newly authorized score-improvement goal. Training allowed within predeclared measured caps; submissions prohibited. Direct composer reproduction/derivative and native pilot remain completed evidence, not score gains.
- Goal: improve beyond the supplied Summit public-code result; preserve own-model controls and completed feasibility evidence.
- Evidence/hypothesis: exact Summit CSV/source recovered, but not yet improved. Related181-member pool already includes TabPFN/TabM, so a family change alone does not establish new signal. [Corrected baseline task](plans/codex-summit-improvement.md).
- Branch: `agent/codex/bounded-tabpfn`; isolated worktree `.worktrees/codex-tabpfn`.
- Ownership: this entry/project-state update, `docs/plans/codex-bounded-tabpfn.md`, [Summit audit](plans/codex-summit-audit.md), [Summit improvement plan](plans/codex-summit-improvement.md), [direct composer adaptation](plans/codex-summit-composer-adaptation.md), [paired Kumo contract](plans/codex-kumo-paired.md), [CatBoost fallback contract](plans/codex-catboost-fallback.md), linked ledger evidence and task-specific local artifacts; explicit handoff before subagent edits.
- Plan: [codex-bounded-tabpfn.md](plans/codex-bounded-tabpfn.md).
- User-requested direction review: [Council3](plans/codex-council3-direction.md), five independent advisors/five anonymous reviews/standard chairman; no furtherAstra(2/2used). Finish currentCatunchanged; then one explicitly bounded canonical128Kumo continuation after identity/costcontract,8h cumulativeincludingv1,90additionalminutesfirstpair. CompletequalifiedCat joins latercontrolregardlesspointestimate; no newLightGBM/batchchangesnow. This is queued, not a current launch or scoreclaim.
- Replay-fix steering: owner explicitly requested “fix it”; [root-cause evidence, fixed-dispatch contract and verification](plans/codex-tabpfn-replay-fix.md). Root owns this plan plus linked ledger/strategy updates; isolated implementation was reviewed and handed back before integration.
- Job owner/monitor: root Codex only. Current private`wasidyms/s6e10-codex-catboost-ctr-v1/1`, kernel140747734, RUNNING after08:54:54UTC2026-10-25singlelaunch; remotehash/privacyreadbackconfirmed. Caps1800perfit/3600first/21600total; no duplicatewhileactive. Kumo140724253/v1TERMINAL firstpairtimeout3600.0076seconds, both21nativecheckspassed/noOOF; partialevidencepreserved. Quota beforeCatlaunch28.88188hours; submissions0.
- Verification: specification PASS and quality APPROVE at code 336d971; each independently reran36 targeted tests. Integrated code 528072f fresh source-only suite: 84 passed, 66 existing warnings, 13.63s. Stable package generated/compiled locally with source/metadata hashes recorded. No new GPU job or submission during the software fix. Earlier recovery matched both incumbent ledger AUCs; failed pilot evidence preserved.
- Verification/handoff: unchanged Summit reproduced exactly; Personalcontextarm rejected(-0.000044486primary). Secondcouncil/Astra2of2 selectedKumo/fallback. Kumo nativeaccepted/resourceinfeasible; no qualitymetric calculated. CatBoost runtimeb41e03c specPASS/qualityAPPROVE,19targetednativeCPUchecks; integratedb58d5d7freshsource-only169passed/66existingwarnings/32.89seconds. Frozen11control excludes incompleteKumo; source/control/packagehashespublished. CatGPU140747734/v1nowrunning, rootmonitors; nativeGPU/OOF/scoregatespending. No higherpublicscore; submissions0, goalACTIVE/unfulfilled.

### Codex TabPFN runner — Codex / GPT-6

- Status: COMPLETE fixed FP32 correction implementation after reported native autocast batch-size replay failure; both replay checks/tolerances unchanged. Root review/launch pending; no job/download/submission here.
- Goal: implement a bounded fold-local raw-feature feasibility pilot and private Kaggle packager.
- Evidence/hypothesis and approach: synthetic TDD for first-fold/context selection, categorical isolation, strict outputs and supervised total deadline; no score claim.
- Branch: `agent/codex/tabpfn-runner`; isolated worktree `.worktrees/codex-tabpfn-code`.
- Ownership handed back to root after FP32 correction verification: new `train_tabpfn.py`, `package_tabpfn_pilot.py`, `test_tabpfn.py`, `test_tabpfn_pilot.py`, this registry entry and [runner plan](plans/codex-tabpfn-runner.md), by explicit root handoff.
- Experiment contract/job owner: none here; no downloads, training, GPU jobs or submissions by this agent.
- Verification: read coordination, execution plan and onboarding; clean task worktree; fetched origin and fast-forwarded from parent task branch (new branch initially had no upstream).
- Verification: baseline 48 tests passed; 22 new synthetic tests passed. Final fresh source-only archive: 70 tests passed, 66 existing warnings, 13.66s at code checkpoint `82b586c`. No model/data downloads, real training, GPU job or submission.
- FP32 verification: 23 targeted tests passed; fresh source-only archive at code `866e15b` passed 71 tests with 66 existing warnings in 10.02s. Dedicated FP32 kernel ID/title and fixed 1,710s cap; native FP32 numerical gate remains unverified.
- Next: root review, new prelaunch contract publication and separately owned bounded FP32 pilot. No full OOF or score-improvement claim.

### Codex stable replay implementation — Codex / GPT-6

- Status: IMPLEMENTED; independent root review pending; fresh source-only verification passed; no model job/download/submission authorized here.
- Goal: repair native query-shape drift with one canonical 128-row padded dispatch, preserving all tolerances/model settings.
- Evidence/hypothesis: repeated same-shape FP32 predictions passed while 64/1000 shape replay failed; fixed native execution shape should remove shape-dependent arithmetic if row/slot independence holds.
- Branch: `agent/codex/tabpfn-replay-fix`; isolated worktree `.worktrees/codex-tabpfn/.worktrees/codex-tabpfn-replay-fix`.
- Owned files by explicit root handoff: `train_tabpfn.py`, `package_tabpfn_pilot.py`, `test_tabpfn.py`, `test_tabpfn_pilot.py`, this entry and [implementation plan](plans/codex-stable-replay-implementation.md). Root retains scientific ledger/root plan ownership.
- Approach: synthetic RED/GREEN, universal fixed dispatch, native anchor/subset/permutation/slot gates and provenance; private 1500-second packager only.
- Verification: baseline71passed; targeted final36passed, prior full81passed/66existingwarnings; fresh source-only archive at code `336d971` passed84tests/66existingwarnings/11.89s. Synthetic RED/GREEN reproduces shape drift, rejects true model/slot/composition drift and validates receipt/packager contract. Initial branch had no tracking upstream, so `git pull --ff-only` reported no tracking information after fetch. Latest origin/main coordination read; root branch already contains main.
- Handoff: parent will perform specification/quality review and exclusively own any native launch. No GPU job, dataset/model download, training or submission here.

### Codex public context implementation — Codex / GPT-6

- Status: REVIEW CORRECTION IMPLEMENTED; independent root re-review pending; synthetic fits only.
- Goal: implement the frozen within-original-fold control/context experiment with strict input validation, complete receipts and unchanged numerical gates.
- Branch: `agent/codex/public-context-stack`; isolated worktree `.worktrees/codex-public-context`.
- Owned by explicit root handoff: new `stack_public_context.py`, `test_public_context.py`, this entry and [implementation plan](plans/codex-context-implementation.md). Root retains the scientific contract, ledger and all real fits.
- Approach: synthetic RED/GREEN tests for isolation, provenance, metrics, bootstrap, inference and failure/export gates; no new dependencies or existing runner changes.
- Verification: clean initial worktree; fetched origin and read latest main coordination. Initial pull reports no tracking branch; branch starts at frozen-contract commit `1cd0b17`.
- Verification/result: targeted37passed; code `5546ac7` fresh source-only full suite121passed/66existingwarnings/21.65s. Strict manifest/identity/probability gates, within-original-fold paired metrics, JSON replay, cap receipts and conditional promotion implemented. Root review corrections preserved centroid ranks and nonconstant threshold.
- Review correction: failed export paths quarantine candidate/partial bytes as `.INCOMPLETE.csv`, preserve scientific evidence and remove stale success claims; four RED/GREEN failure cases, targeted41passed; fresh source-only full125passed/66existingwarnings/29.86s.
- Handoff: independent root re-review before any real run. Root will assert frozen top manifest SHA in launch wrapper. No GPU, download, real-data fit or submission occurred; no scientific gain claimed.

### Codex Kumo implementation — Codex / GPT-6

- Status: REVIEW CORRECTION IMPLEMENTED; native class-label alignment handed back to root at `6c518da` for re-review. No native fits/downloads/jobs/submissions here.
- Goal: frozen paired raw/FE native Kumo runner with context-only statistics/schema and strict complete-output/replay/resource gates.
- Branch: `agent/codex/kumo-pair`; isolated worktree `.worktrees/codex-kumo-pair`.
- Ownership: `train_kumo_pair.py`, `package_kumo_pair.py`, `test_kumo_pair.py`, this unique entry and [implementation plan](plans/codex-kumo-implementation.md). Root retains scientific contract/ledger and launch ownership.
- Approach: synthetic TDD; reuse fixed128 dispatcher and existing supervisor; lazy native imports; preserve partial receipts; no candidate export.
- Verification: published/read-back claim and code through `9f3da25`; 18 targeted synthetic tests passed. Class-label correction `6c518da`:25targetedpassed; fresh source-only full suite150passed/66existingwarnings/30.55s. Read AGENTS/strategy/ledger/onboarding/latest origin/main; initial branch no upstream, then tracking established. Git metadata uses authorized escalated writes.
- Handoff: checkpoint digest supplied by root; native CUDA feasibility remains unverified. No model install/download, real-data fit or submission authorized for this agent.

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
- Job owner/monitor: root Codex only. No new job yet; both allowed Astra calls already used.
- Verification/handoff: council4 complete5advisors/5anonymousreviews/chairman, noAstra remaining. [Frozen Kumo continuation](plans/codex-kumo-continuation-contract.md): same canonical128/rawFE, qualifiedCatcontrol, gates14vs13and14vs12;24900newworkerseconds/5400firstpair/25%projection/8cumulativeh. Regenerateallpairs,no reuse. Freshquota27.82529h/noactivejobs22:54UTC2026-10-26. Isolatedbranch/worktree ready, baseline169pass. Explicit runtime-only implementation handoff; root retains launch/monitor/scientific decisions. No new GPU job yet.

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

- The owner explicitly resumed Codex training/evaluation on 2026-10-26; historical pause records remain preserved. Submission permission was not granted.
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

### Codex CatBoost implementation — Codex / GPT-6

- Status: RECEIPT CORRECTION IMPLEMENTED; root delta specification/quality review pending; synthetic CPU verification only, no real fits/jobs/submissions.
- Goal: fixed source-derived native CatBoost CTR fallback with isolated inner selection, outer refit, immutable teacher input and strict native/resource gates.
- Branch: `agent/codex/catboost-ctr`; isolated worktree `.worktrees/codex-catboost-ctr`.
- Ownership by explicit root handoff: new `train_catboost_ctr.py`, `package_catboost_ctr.py`, `test_catboost_ctr.py`, this unique subsection and [implementation plan](plans/codex-catboost-implementation.md). Root owns the scientific contract/ledger and all launches.
- Approach: synthetic TDD, reuse recipe constants/hash/native comparison and process-group supervision; lazy CatBoost import; preserve partial evidence.
- Verification: read AGENTS, strategy, experiments, onboarding and latest origin/main; clean initial branch at `1a78a80`; fetch succeeded; initial pull reports no tracking branch.
- Verification checkpoint: runtime `47e6af1` published/read back;19targetedchecks passed including actual tiny native CPU CatBoost1.2.8 CBM replay. Fresh source-only full suite169passed/66existingwarnings/32.66seconds. Generated source closure imports outside the repository. Latest main`9dfbfc8` registry update integrated deliberately; no runtime changes.
- Receipt correction: RED reproduced stale selection.json self-hash; own receipt now excludes itself, with every listed hash checked against disk and final run metrics independently pinning the completed selection file. Runtime correction`b41e03c`published/read back; targeted19passed, fresh source-only full169passed/66existingwarnings/33.02seconds.
- Handoff: root delta specification then quality review before any native GPU run; no launch by this agent.
