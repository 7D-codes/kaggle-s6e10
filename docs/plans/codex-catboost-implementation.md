# CatBoost CTR implementation

Software-only implementation of the [frozen scientific contract](codex-catboost-fallback.md). Root is sole launch owner; no real-data fit, GPU smoke, job or submission here.

## Scope and strategy

Own only train_catboost_ctr.py, package_catboost_ctr.py, test_catboost_ctr.py and this plan/own registry entry. Reuse recipe feature constants, immutable teacher validation patterns, comparison/hash functions and existing process-group supervisor. No existing runtime changes. New packager builds/compiles only.

Implement fold-local pure JSON preprocessing; inner selection accepts only outer-training rows/labels, then fresh outer preprocessing/model with exact selected count. Require native CPU CBM replay, complete five-fold coverage and fail-closed deadlines. Frozen upstream teacher input remains required; no historical teacher regeneration claim.

## Verification and progress

Initial repository clean; read required coordination/latest origin/main. Fetch succeeded using authorized Git metadata access. Initial pull has no tracking branch. Next: publish claim, add failing synthetic tests, implement and run targeted/native CPU/source-only full suite; publish code for root specification then quality review.

## Handoff

No training launch authorized here. GPU behavior/performance remains unverified until root's separately authorized job.
