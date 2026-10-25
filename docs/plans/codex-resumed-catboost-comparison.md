# Resumed frozen CatBoost ensemble comparison

Owner: Codex / GPT-6. Documentation branch: `main`; execution uses the preserved `agent/codex/bounded-tabpfn` worktree and its task-owned ignored artifacts. On 2026-10-26 the owner explicitly said "GO AHEAD YOU HAVE FULL RIGHTS" after discussion of the training/comparison pause. This resumes training and evaluation for this score-improvement effort. Prior no-submission instructions remain in force. No change to protected AGENTS.md is needed: explicit owner instructions override its default pause.

## Objective, ownership and unchanged contract

Determine whether the already completed integrity-qualified CatBoost member adds ranking information beyond the frozen eleven-member control. Own this plan, Codex's resumed-comparison registry entry and the appended experiment-ledger entry. No runtime code edits, new CatBoost fit or GPU job for this comparison.

Reuse the existing council-selected and preregistered [CatBoost contract](https://github.com/7D-codes/kaggle-s6e10/blob/2e622f690bc78fec8eb5571fed49be64e277f155/docs/plans/codex-catboost-fallback.md). The [independent audit](codex-catboost-readonly-audit.md) proves complete native component outputs; standalone development ROC AUC is 0.9610177594499577. Exact Summit OOF remains unavailable, so an offline gain cannot establish a public improvement.

Frozen execution source: `research/summit-score-goal/analysis/compare_catboost.py`, SHA256 `5928390764f92c5e4e381975620290ae341ca0358cf6654e3a27747f855591b9`. Reviewed helper `stack_public_context.py` SHA256 `b20c9043c05eebe17d5673839312af6e12466db50577f303800ed16e15e010ec`. Frozen control manifest SHA256 `72ab4f3c6dd81f63cbb342f763bfbf01713e79e9a85a5b55c320cd5a00f3729e`; complete native audit receipt SHA256 `27b927899650405a26192419f9fe7ec66cbda527bfbba32be4b22204292b3268`.

Five original shuffled stratified folds seed42; two reversed half-splits within each excluded original fold, half seed20261025+fold. Fixed logit inputs, L2 logistic C1/lbfgs/max_iter2000/tol1e-7, twenty fits (ten per arm). Primary pair-count-weighted ten-cell ROC AUC; pooled secondary. Two hundred paired stratified fixed-prediction bootstrap replicates seed314159, conditional only. Unchanged control must reproduce its saved predictions at original rtol1e-6/atol1e-7. No C/weight/feature/source search.

Gate: primary gain >=0.00002, positive in at least four of five original folds, positive lower conditional95%gain bound, and complete integrity/convergence/coverage. Only a pass exports the fixed UNSCORED Summit rank correction with weight0.05. Failure exports no candidate and earns no rescue tuning. Test-input averaging and historical development/source-assurance caveats remain explicit.

Resource cap: four CPU threads, 10800 seconds with process timeout; no GPU quota consumed. Single output directory `results/summit-catboost-comparison-v1`; fail if it already exists rather than overwrite or duplicate. Stop on integrity, convergence, coverage or timeout failure. Preserve all evidence.

## Progress and verification

Fetched/pulled current main, read current coordination/onboarding documents and inspected the frozen source and completed audit. Main and execution worktrees clean; comparison output absent. Publish/read back this claim before fitting. After completion independently recompute coverage, metrics/gates and coefficient-based predictions from saved outputs; record exact outcome/runtime/hashes here and in the ledger.

## Handoff

If comparison fails, close this CatBoost addition without tuning. Complete integrity-qualified Cat remains eligible in the subsequent council-predeclared Kumo control regardless of its gain. Any GPU continuation needs its concrete published cumulative resource contract, fresh quota/no-active-job check and source/native review. Submissions remain forbidden.

## Measured result and independent audit

The unchanged comparison completed in **47.9543 seconds** and passed every frozen gate. Primary development ROC AUC: control **0.9616869424637373**, plus-Cat candidate **0.9617919079329712**, gain **0.00010496546923383286**. All five original-fold gains are positive: 0.00005782429, 0.00011800100, 0.00008761002, 0.00012774616, 0.00013364573. Conditional fixed-prediction bootstrap95%gain interval **[0.00007762420, 0.00014243317]**; 200replicates/seed314159. This is conditional development evidence, not an untouched or public score.

Independent audit verified all699635IDs/targets/folds/cells and one-time coverage; all20coefficient states, split-index hashes, original-fold exclusion and convergence; recomputed primary/per-fold/pooled AUC and gates. Coefficient-based replay exactly matches every saved OOF and test meta prediction (maximum error0). Original eleven-member control also reproduced its prior predictions. Full299844-row candidate exactly follows the predeclared0.05rank-correction formula on saved averaged-meta inputs; original Summit file/hash preserved. Bootstrap algorithm/seed/gate inspected, not redundantly rerun.

Candidate: `results/summit-catboost-comparison-v1/candidate-catboost-UNSCORED.csv`, SHA256 **93282e7daa51ffa02f411fa186324edbb83143df49790933a6415087721cbd6b**. Comparison metrics SHA256 `a052c32c500db64ccbac5cb6a80024dd4c5df774e831e82231659241fcf06fd4`. Local independent receipt `reports/summit-score-goal/catboost-execution/independent-comparison-audit.json`. These paths are in the preserved bounded-tabpfn worktree; no raw predictions enter Git.

Decision: retain the qualified unsubmitted candidate. **Do not add the offline gain to Summit's reported0.96213 or claim a new public score.** Exact Summit OOF is missing, public producer assurances remain conditional, and test inputs are averaged before meta inference. Actual public>0.96218 remains unverified. Submissions0; no GPU/Cat training spent in this comparison. Runtime Python3.13.11/NumPy2.5.4/pandas3.1.0/sklearn1.9.1/SciPy1.18.1, four CPU threads.

Ownership released for this completed comparison. Next: the existing council's one bounded canonical128Kumo experiment, with Cat retained in its frozen control regardless of this positive outcome. No rescue tuning or broader search is triggered by the pass.
