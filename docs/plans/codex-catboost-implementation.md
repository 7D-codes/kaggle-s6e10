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
