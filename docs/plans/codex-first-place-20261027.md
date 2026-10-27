# First-place campaign — 2026-10-27

## Goal, boundary and ownership

User replaced the previous goal with “let’s take the first place” and activated it. Success requires authoritative evidence of first place for the user's eligible submission on the relevant leaderboard; offline scores or an unsubmitted CSV cannot establish rank. Track public placement and eventual private final result separately. No promised score or rank. The earlier explicit no-submission instruction remains in force. Repository instructions pause fitting until explicit resumption; one clarification is pending while read-only work proceeds.

Root Codex owns this plan, its strategy entry and linked ledger evidence on main. No new runtime source implementation is currently proposed. Reviewers are read-only. Preserve all original predictions and audit evidence.

## Current evidence and next actions

Main synchronized at45ee4956a1693a106c1de4f978fa6cb23b79f879. Existing Kumo continuation completed; the formerly stalled read-only observer subsequently captured all eligible outputs and reports successful artifact audit at05:35:44UTC. Revalidate against actual files rather than trust the receipt alone. Prior full CatBoost surrogate gain is +0.00010496547; existing correction analogue retains +0.00000690542. Neither establishes Summit/public gain.

1. Obtain fresh primary leaderboard and competition metadata, quantify public score gap without inferring private ranking.
2. Independently verify completed Kumo source/context/IDs/folds/prediction aggregation/metric evidence and confirm frozen comparison input readiness. No fitting in this step.
3. Upon explicit evaluation resumption, execute the already-reviewed frozen thirty-fit twelve/thirteen/fourteen-member comparison once, unchanged control/configuration/gates, four CPU threads and10800second cap. No new GPU run is needed.
4. Evaluate the complete candidate recipe, preserve the full challenger separately from the small Summit overlay, and determine whether a new complementary model is warranted. Any new fitting contract must precede execution; no automatic weight or feature sweep.
5. Any eventual submission requires separate explicit authorization and exact candidate provenance. Only a returned scored submission/rank can establish public placement; never substitute an offline milestone for this goal.

## Progress classification

This is the first turn of the newly activated goal. The preceding question answered overfitting conceptually; it was not an execution turn of this goal. Current initial state inspection has found actual completed artifact audit files previously unverified in chat. Goal remains active, not complete.


## Explicit resumption and next execution contract

Owner answered **“Resume training and evaluation; no submissions.”** This resumes fitting and GPU work within declared experiment/resource contracts; it does not authorize competition submissions. No further resumption clarification is needed.

Next run is the already-reviewed frozen comparison from `agent/codex/kumo-comparison-20261027`: source SHA256 `340fb7272f5e07610b6cba2699d1b30b80873c9f8f6fdcec95862d7f4043e66c`, helper `b20c9043c05eebe17d5673839312af6e12466db50577f303800ed16e15e010ec`, control manifest `91a6fac84d243482579d18ef5c2ab52b33b6ecf5ad22c22c457bcddf094d628d`. Twelve unchanged members versus +rawKumo versus +raw+engineeredKumo. Thirty within-original-fold meta fits,5foldseed42/halfseed20261025+fold, logisticC1/lbfgs2000/tol1e-7, fourthreads and10800CPU/wallsecond cap. Primary14vs13 AND14vs12 gain≥0.00002,≥4/5positivefolds and lower conditional200bootstrapseed314159interval>0;13vs12descriptive. Reproduce historical twelve-member OOF/test at originalrtol1e-6/atol1e-7 before accepting. Existing0.05export and all historical gates unchanged. Root sole executor; no retry or output overwrite without diagnosing failure.

Run after independent source/input readiness using the existing scientific environment and `stack_kumo_continuation.py --run --root /home/toro/Projects/kaggle-s6e10/.worktrees/codex-tabpfn`. Output `results/summit-kumo-comparison-v1/` beneath that root. Existing reviewed code was already tested37targeted/216fresh-source-only; current source hashes must match before relying on those checks. This CPU comparison does not train a new base model or consume GPU quota.


## Verified leaderboard and Kumo result

Kaggle API snapshot2026-10-27T05:51:14UTC: leader Masaya Kawamata0.96427, second0.96352, third0.96276, team Token it easy - LAB7 fourth0.96254. Competition metadata independently reports userRank4, deadline2026-10-31T23:59UTC,10daily submissions andmaxteam3. Team submission57687430 at03:22:53UTC scored0.96254; its description explicitly identifies a public-label-posterior/LB-fitted display-only result and says not to select it. This was submitted by a teammate, not this agent. Preserve the recorded description; do not treat it as validated generalization. Display-score gap to first is0.00173; gap from Summit0.96213 is0.00214. No inference about leaders' methods follows from their scores. User preference on public-first versus final-private-first is pending. No new submission occurred.

The completed Kumo artifacts passed independent full ID/label/fold/context/preprocessing/hash checks. Every saved per-fold output reconstructs fullOOF and test means within3e-16; source/embedded-code identities checked;210remote native receipt checks pass at original tolerances. Native Kumo computation was not rerun locally. Final account quota receipt at05:35:41UTC shows22.76730194GPUhours remaining and21814.756cumulativeKumo seconds upperbound, within the original8hourcap.

Root ran the reviewed frozen comparison once, exit0,74.05123wallseconds/110.54534CPUseconds. Twelve-member control andtest replay errors both0.0. All30savedcoefficient states independently reproduce everyOOF/testprediction exactly; complete hash/ID/fold/cell/metric summaries rechecked.

| Contrast | Pair-weighted development AUC gain | Positive original folds | Conditional95% gain interval |
|---|---:|---:|---|
| FE14vs13 | +0.00001063413889920206 | 3/5 | [0.00000242448,0.00001904416] |
| Joint14vs12 | +0.000007418444824559567 | 4/5 | [-0.00000316038,0.00001844126] |
| Raw13vs12 descriptive | -0.000003215694074642492 | 2/5 | not computed |

Decision: **reject promotion; close this Kumo experiment.** Both mandatory gains are below0.00002; FE also fails≥4positivefolds and joint interval includeszero. This is not a precise null. No candidate CSV was exported, no GPU retraining/retry and no submission. Retain the unchanged twelve-member control at0.9617919079329712; joint14is0.9617993263777959. All are conditional development estimates, not public scores. No bootstrap replication was needed to confirm the independently failed gain gates.

Artifacts remain ignored at `.worktrees/codex-tabpfn/results/summit-kumo-comparison-v1/`. OOF SHA256 `b6e41104ec3d0cc4706f9d5bb515831241a3ef715e4513d8a6c99d2202fed230`; test `a5aade9b53c3492a3cbb8d4da4acb68bed416a9cf0a6bf502df2dd7065ed5538`; coefficients `2e97b8cc116a133e41e510de497d9b65975b030f4cebef8d67ebdc10f3f79154`.
Metrics SHA256 `f94936754288b39a98e391ffed0849fd8e0e8c9840e291597443f337857f376b`. RuntimePython3.13.11/NumPy2.5.4/pandas3.1.0/sklearn1.9.1/SciPy1.18.1; originalfoldseed42/metahalfseed20261025+fold; source/config/control/gates unchanged from the predeclared contract.
