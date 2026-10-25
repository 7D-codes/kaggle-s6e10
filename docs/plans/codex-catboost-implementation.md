# CatBoost CTR implementation

Software-only implementation of the [frozen scientific contract](codex-catboost-fallback.md). Root is sole launch owner; no real-data fit, GPU smoke, job or submission here.

## Scope and strategy

Own only train_catboost_ctr.py, package_catboost_ctr.py, test_catboost_ctr.py and this plan/own registry entry. Reuse recipe feature constants, immutable teacher validation patterns, comparison/hash functions and existing process-group supervisor. No existing runtime changes. New packager builds/compiles only.

Implement fold-local pure JSON preprocessing; inner selection accepts only outer-training rows/labels, then fresh outer preprocessing/model with exact selected count. Require native CPU CBM replay, complete five-fold coverage and fail-closed deadlines. Frozen upstream teacher input remains required; no historical teacher regeneration claim.

## Verification and progress

Initial repository clean; read required coordination/latest origin/main. Fetch succeeded using authorized Git metadata access. Initial pull has no tracking branch. Next: publish claim, add failing synthetic tests, implement and run targeted/native CPU/source-only full suite; publish code for root specification then quality review.

## Handoff

No training launch authorized here. GPU behavior/performance remains unverified until root's separately authorized job.

## Runtime checkpoint

Implemented the fixed 51-feature/21-category component and build-only private packager. Inner selection uses only its 80% outer-training split; outer refit uses fresh preprocessing, exactly selected iterations, and no eval_set/early-stop argument. Qualified teacher hashes/provenance/order precede native construction. Native CPU generation/reload/shape/permutation/JSON replay uses unchanged rtol1e-6/atol1e-7; CPU training fallback is rejected. Atomic stage deadlines cannot be cleared after expiry; supervisor polls at most one second and kills the existing whole process group for phase/first/total caps. Full outputs require five folds and exact IDs/labels/order; interrupted runs retain partial metrics/predictions. Source default value_counts/groupby excludes missing keys; missing numerical counts/routes use zero/fitted fallback, while raw NaNs are retained.

RED observed missing runner/packager, and explicit missing-value-count and expired-phase transition regressions. Targeted 19 tests passed, including a native CPU synthetic CatBoost1.2.8 CBM roundtrip. Installed only the CatBoost1.2.8 wheel with --no-deps into existing /tmp/s6e10-test-env; no GPU toolkit/model weights. Full workstation suite passed169tests/66existingwarnings/33.04seconds before final missing-key correction; targeted19passed afterwards. Generated package closure imports from an empty source directory. Fresh source-only full verification follows this code checkpoint.

Native training GPU support, real-data runtime/score and historical teacher regeneration remain unverified. No real-data fit, GPU smoke, remote job, download of model weights or competition submission occurred. Root specification then quality review required before any launch.

## Verified handoff

Runtime checkpoint `47e6af18604aae35a547890cb2ed3e2402350bf8` published and remote SHA read back. Fresh source-only archive of that commit, with working directory `/tmp/s6e10-catboost-source-check`, passed169tests with66existingwarnings in32.66seconds. Targeted19tests passed in2.82seconds. Native CPU test is a tiny synthetic12-tree fit, not a real-data experiment. Package generated/compiled and its complete embedded import closure tested from an empty directory.

Fetched latest origin/main`9dfbfc8`; inspected its only change (Airline Handoff registry completion/release links) and deliberately merged it at`2dfa005`, preserving both that entry and the CatBoost entry after an append-context conflict. Runtime files unchanged by merge. Source/staged diff checked for whitespace, credentials and restricted artifacts; only the three new runtime/test files and public-safe coordination evidence belong to this task.

Handoff to root for independent SPEC then QUALITY review. No real-data fit, GPU smoke, remote job, competition submission or model-weight download occurred; no scientific score improvement is claimed. Native GPU operation/performance and full production inference remain unverified. Training/submission/launch authority stays with root under the frozen contract.

## Receipt self-hash correction

Root explicitly returned runtime ownership for one metadata correction before launch. RED: the synthetic inner-selection test found selection.json listed its pre-rewrite bytes in its own hashes. Minimal fix excludes selection.json from that receipt's own hash map; every remaining listed artifact must match disk. The full-run synthetic success check separately requires final metrics to hash each completed selection.json externally. No model, tolerance, fit settings, launch or submission changes. Targeted19tests passed in2.82seconds. Published runtime correction `b41e03c472ae58abf03ac784a55c7fd1b0879c2f` and verified matching remote SHA. Fresh source-only archive of that exact commit, executed from `/tmp/s6e10-catboost-receipt-source-check`, passed169tests/66existingwarnings in33.02seconds. Root SPEC then QUALITY delta review pending; no real-data fit, GPU smoke, launch or submission occurred.
