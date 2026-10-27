# Conditional resource decision council

Owner: Codex / GPT-6. Branch: `main`, advisory research only. Owned files: this plan, its strategy subsection and linked ledger entry. No runtime edits, fits, new jobs or submissions. Latest owner-supplied AGENTS pauses training/evaluation; permission request for the already prepared CPU comparison remains unanswered. Both allowed Astra calls have been consumed; this council uses standard advisors only.

## Framed question

If the sole current Kumo run ends in a resource/projection failure, should the next proposed experiment exploit both T4 devices to complete the same raw/engineered comparison, or should we abandon that resource path and propose a distinct architecture/representation? This is a conditional recommendation, not permission to restart or relax the current contract. If the run completes or fails quality rather than resources, identify the corresponding decision instead.

Goal: materially stronger predictions than reported Summit public ROC AUC0.96213, targeting above0.96218; no submissions. Exact Summit OOF is unavailable, and no offline gain proves public improvement. Qualified CatBoost adds0.00010496547 development ROC AUC to the frozen ensemble, five positive folds, but its fixed candidate is unscored.

Live private Kumo kernel141062725/version1 began2026-10-26T23:05:27UTC and was RUNNING with nofailuremessage at23:58:45UTC. Model is KumoSmall,100kfold-onlycontexts,2estimators,fp16,raw21versus29features,canonical128querypadding,unchanged21nativechecks/arm at rtol1e-6/atol1e-7; all5pairs required. Control12 includes Cat. Primary14vs13 and mandatory14vs12 each require gain>=0.00002,>=4positivefolds and positive lower conditional bootstrap bound. No partial promotion, raw-only promotion, weight tuning or first-fold quality selection.

Resource contract: eight cumulative Kumo GPU hours including old v1's3606roundedseconds; current worker24900seconds plus294reserve, firstpair5400seconds, and measuredcomplete-workprojection6.25*firstpair<=24900. Old v1 timed out at3600seconds with no full pair/OOF; both arms passed native checks, raw predictions and FE validation finished, FE test unfinished. This was not a quality rejection. All outputs preserved; current run regenerates every pair. Any successor proposal must count BOTH previous runs' actual consumed quota, keep within a newly explicit cumulative budget, and use a fresh quota/no-active-job check. No automatic credit for partial artifacts or budget extension.

New evidence: current source forces `CUDA_VISIBLE_DEVICES=0` in supervisor AND runner; previous Cat receipt independently lists two T4s of15360MiBeach. Current live unused-device availability is unobserved. Kaggle current specifications list two T4s, fourCPUcores and29GBhostRAM. Official announcement says both GPU environments draw from the same quota, without explicit current per-device rate. Unsloth's own CI source reports wall-clock session charging and independent payloads on two cards. Our account's dual-use charging and speedup remain unmeasured. Independent arm processes might reduce walltime without changing model/queryshape, but sharedCPU/RAM, GPUmapping, RNG isolation, artifacts/aggregation/deadlines and native equivalence require qualification; VRAM does not pool. No implementation exists.

Recent public source review found no new qualified member: larger standalone TabPFN gave no incremental large-pool gain in another author's study; broad38membermetaCV0.962021/public0.96169 used conditional/nonnested baseOOF. Already rejected own epochs/applicability/teacher/Personal-interaction arms must not be recycled. TabM remains a lower-evidence untested architecture, not a selected successor.

Each advisor must choose a conditional recommendation, specify the minimum missing evidence and stop rule, and separate resource feasibility from predictive improvement. Do not invoke external experts/Astra, launch anything, fit models, or edit files. Read AGENTS/STRATEGY/EXPERIMENTS; root synchronized coordination immediately before this task. Three available child slots require independent3+2waves, then anonymous peer review and chairman synthesis. Do not let early responses enter later independent prompts.

## Progress and verification

Ownership claim published and remotely verified at `750f124d119da5ab62310aaa9e35dfd11f18ca9d` before advisor work. Five independent standard advisors completed, followed by five new anonymous reviewers and a standard chairman. Three child slots required overlapping3+2waves; independent prompts did not include earlier responses. No additional Astra calls. Anonymous mapping was A=Outsider, B=Expansionist, C=Contrarian, D=Executor, E=First Principles. Reviewers preferred E twice, B twice and C once; four flagged D's implementation-cost assumption, one E's insufficiently explicit resumption condition. Root preserved the full advisor packet and review findings in ignored research storage; the public-safe decision and material disagreements are below.

## Where the council agrees

A resource failure leaves predictive value unresolved; faster hardware does not establish signal or favorable quota use. Preserve the current run, model/context/features, canonical128 queries, native tolerances, complete coverage and both quality gates. All new qualification/fits/jobs remain paused. Both previous runs' consumption must count.

## Where the council clashes

All advisors conditionally preferred two-device qualification to immediately selecting a weakly supported architecture. Peer review challenged that shared framing: affordability is necessary but does not prove best expected return. The executor's narrow scientific comparison still needs substantive engineering for device mapping, host memory, artifact isolation, aggregation and cleanup. Chairman therefore deferred successor selection rather than endorsing a retry. Another architecture remains uncertain; TabM is unverified, while full context is a distinct information change rather than more estimators or seeds.

## Blind spots caught in peer review

- Original ceiling is **28800 cumulative quota seconds**, not another eight hours:3606prior+24900current+294reserve. Current projection makes the first completed pair<=3984seconds, stricter than its separate5400secondcutoff. Any future qualification/regeneration/aggregation/provider overhead counts against what actually remains after BOTH runs.
- Parallel time follows the slower arm plus overhead, not half the serial sum. Startup, unequal arm duration, fold variability, four shared CPU cores and combined host memory must enter a conservative projection. Current-account charging remains unmeasured for concurrent use.
- Native tolerances alone do not prove equivalent computation. Require seeded, nonconstant raw-row replay against declared references, with unchanged canonical query shape and isolated artifacts; no tolerance relaxation.
- Declare engineering and qualification limits, alternative use of quota, and whether qualification outputs are eligible before spending. Assume no reuse credit by default; do not pay twice or retrospectively promote partial outputs.
- Reused development labels and adaptive selection create uncertainty outside the conditional bootstrap. Passing the gates does not prove Summit/public improvement. Do not repeat the impossible exact-Summit-OOF recovery search; only owner-provided artifacts could change its known absence.

## Recommendation

**Defer successor selection until terminal receipts permit a numerical contract.** Do not modify, cancel or restart the current run to test dual-device work.

If the exact version fails solely on resources/projection, reconcile actual consumption, arm timing and memory for both runs and retain reserve. Reject the two-device path before engineering if even optimistic overlap cannot fit the remaining original ceiling. If plausible, compare bounded implementation/qualification cost with stopping and preserving quota for a separately contracted representation experiment. Missing charging evidence prevents full continuation. Qualification failure ends this resource path; no automatic rescue, cap extension or smaller-context substitution. Any qualification still requires explicit resumption and a fully numerical contract; this council does not supply one while terminal values are unknown.

If the current run completes, audit integrity and full outputs; run the frozen14vs13and14vs12comparison only after explicit evaluation resumption. A quality failure retires this configuration. An integrity failure requires diagnosis before scientific conclusions. Passing produces only a conditional offline candidate, not an observed leaderboard gain.

## One first action and handoff

Read-only live quota checkpoint at2026-10-27T00:10:32UTC: account used3.258856404722222GPUhours, remaining26.741143595277777hours, increase since launch1.0841461269444443hours. Thus usage is being reported during this running session; it is not necessary to wait for termination to observe some consumption. This is account-wide/provisional, not final per-job attribution, dual-device charging proof or successor budget qualification. Exact version remained RUNNING with nofailuremessage at00:10:09UTC. Live output inventory at00:08:08UTC had zero files, no persisted log and no next page; no fold-completion inference follows.

Obtain the existing version's terminal receipt and reconcile actual consumed quota, arm-stage timings and memory evidence. Until then, monitor that exact live version without restarting. Latest observed API status remained RUNNING with an empty failure message at2026-10-27T00:03:46UTC. No training, qualification, benchmark, runtime edit, job launch or submission occurred in this council. Root checked documentation scope, local links, whitespace and credential patterns; no runtime tests were needed because runtime code is unchanged. Ownership released after publication. The improvement goal remains unfulfilled.
