# Frozen paired Kumo implementation

Root contract: [paired Kumo](codex-kumo-paired.md). Owned runtime files: train_kumo_pair.py, package_kumo_pair.py, test_kumo_pair.py. No real-data fit/model installation/download/job/submission in this task.

## Scope and approach

Use native Kumo small classification with fixed two-estimator cached inference and universal fixed128 dispatch. Pure feature state stores context categories/route statistics/cross-fit TE; native schema remains frozen to four categorical columns. Both arms share selected context rows. Package embeds existing dispatcher source and a process-group supervisor with setup-inclusive first-pair and total deadlines. Fail closed for hashes, identity, incomplete coverage and native replay drift.

Callers: new runner CLI and private package only. Existing source-only tests must stay independent of torch/SDM. New tests cover feature isolation, categories, TE exclusion, paired selection, replay, deadlines, hash/ID/coverage gates and package compilation.

## Progress and verification

- Read project coordination/onboarding and native source pinned by root. Git fetch requires escalated metadata writes and succeeded; initial task branch has no upstream.
- Ownership claim published/read back at `1768518`; code checkpoints `82c02ad`, `23b681f`, and early-native-gate correction `9f3da25` published/read back. Root requested independent specification then quality review.
- Synthetic RED reproduced missing modules, StringTensor JSON archive mismatch, nonfinite transform receipt serialization, and late replay ordering (`[1200,300]` calls before replay). Final targeted checks: 18 passed in 1.20s. Early gate now makes only the 1000-row calibration call before forced rejection; no full outputs/marker.
- Existing full suite initially passed133tests; fresh source-only archive at `23b681f` passed142tests,66existingwarnings,30.70s. Final `9f3da25` fresh source-only archive passed143tests,66existingwarnings,30.91s (`/tmp/s6e10-kumo-final-R1nGHc`, local regenerable verification only).
- Verified strict context/category/TE isolation, shared selection, fixed128 drift rejection, native missing-value transform equality, complete coverage/IDs, fail-before-construction hashes, both process-group deadlines, package pins/private metadata/compile, failed/partial receipts and marker ordering. Prediction tolerance unchanged (`rtol=1e-6,atol=1e-7`); preprocessing-only comparisons allow aligned missing values.
- Reused `train_tabpfn` dispatcher/replay and `package_tabpfn_pilot.SUPERVISOR`; SDM/torch imports remain lazy. Package guards the exact109936978bytes/SHA and immutable revision/tag; runner uses unmodified native architecture/cache with official weights-only state load. No model dependencies installed here.
- Native runtime/checkpoint loading and GPU feasibility are deliberately unexecuted; independent root reviews and launch ownership remain required.

## Handoff

Runtime files handed back to root at `9f3da25` for independent specification then quality review. Clean code checkpoint and remote SHA read back. Staged whitespace and credential-pattern checks passed; only source/tests/public-safe coordination docs committed. No model install/download, real-data fit, GPU job, remote launch or competition submission occurred. Native GPU calibration, output feasibility and timing remain unverified.
