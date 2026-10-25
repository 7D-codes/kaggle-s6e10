# Shared Strategy and Agent Ownership

## Project state

**Historical Stable Pilot COMPLETE; native numerical gate PASSED.** Current training authority is the active-goal instruction below. Competition submissions remain prohibited.

**Latest active-goal authorization: training RESUMED for root Codex's score-improvement work using available Kaggle quota; submissions prohibited.** The owner requires llm-council at major decisions and allows at most two Astra calls. [Current owned goal/evidence](plans/codex-summit-score-goal.md). Fresh quota29.88344hours remaining, no active jobs. Original objective remains unfulfilled; an unscored file is not a gain.

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
- Ownership: this entry/project-state update, `docs/plans/codex-bounded-tabpfn.md`, [Summit audit](plans/codex-summit-audit.md), [Summit improvement plan](plans/codex-summit-improvement.md), [direct composer adaptation](plans/codex-summit-composer-adaptation.md), [paired Kumo contract](plans/codex-kumo-paired.md), linked ledger evidence and task-specific local artifacts; explicit handoff before subagent edits.
- Plan: [codex-bounded-tabpfn.md](plans/codex-bounded-tabpfn.md).
- Replay-fix steering: owner explicitly requested “fix it”; [root-cause evidence, fixed-dispatch contract and verification](plans/codex-tabpfn-replay-fix.md). Root owns this plan plus linked ledger/strategy updates; isolated implementation was reviewed and handed back before integration.
- Job owner/monitor: root Codex only. Current private`wasidyms/s6e10-codex-kumo-paired-v1/1`, kernel140724253, launched06:10:23UTC2026-10-25, RUNNING; exact remote source/privacyreadbackpassed. Firstpairedfold1h/total8h; no duplicate/fallback until terminal. Historical stableTabPFNpilot140647663/v1COMPLETE; failedearlierpilotsretained. Quota beforelaunch29.88344hours; submissions0.
- Verification: specification PASS and quality APPROVE at code 336d971; each independently reran36 targeted tests. Integrated code 528072f fresh source-only suite: 84 passed, 66 existing warnings, 13.63s. Stable package generated/compiled locally with source/metadata hashes recorded. No new GPU job or submission during the software fix. Earlier recovery matched both incumbent ledger AUCs; failed pilot evidence preserved.
- Verification/handoff: unchanged Summit reproduced exactly; Personalcontextarm rejected(-0.000044486primary). Secondcouncil/Astra2of2 selected [paired native Kumo](plans/codex-kumo-paired.md) plus fixed CatBoost fallback. Source-only teacher qualified and included in frozen11control regardless of tinynegativecontrast. Kumo runtime6c518da independentlyspecPASS/qualityAPPROVE,25targetedtests; integrated6749334freshsource-only150passed/66existingwarnings/30.69seconds. Sealedprivatepackage launchedoncewithverifiedremotehash/privacy; rootmonitors140724253/v1. NativeGPU/scoreacceptancepending; no higherpublicscore. Submissions0, goalACTIVE/unfulfilled.

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

- Latest owner instruction explicitly resumes score-improvement training using available quota, with predeclared evidence/resource gates and root launch ownership. Submissions remain prohibited; earlier pause statements are historical.
- Current diagnosis suggests limited model diversity; TabPFN is only an unverified research lead. Its existence is not permission to install/download/launch it.
- Preserve historical negative results and the incumbent. No new agent should start a fresh feature/epoch sweep by default.
