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

Ownership claim prepared before advisor work. Council not yet convened; current job unaffected. Publish adviser/peer-review/chairman outcome and exact limitations before closeout. This plan does not authorize a successor.
