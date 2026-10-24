# Direct Summit Composer adaptation

Owner: root Codex. Branch: `agent/codex/bounded-tabpfn`. Latest owner steering explicitly requests building directly on their public code rather than starting another model. This replaces the proposed TabPFN feature-training direction. Training remains paused; no GPU job or submission.

## Concrete task

1. Retrieve only the public prediction files named/hash-pinned by the exact supplied notebook361447722/APIv2. Preserve source notebook and helper locally under ignored `research/`; do not commit copied third-party code or predictions. Respect private-source boundaries.
2. Run the original numerical cells locally, preserving the complete existing recipe. Require the default0.08TabPFNshare/step5 output to match declared ranking SHA256 `21f6fe37dd8f1b7673677d00d6d1bbd75190534d0d557dee76b9fd897a4fdfed` and original exported CSV SHA256 `0c6d65f427725bd57ed2885826c451f6eb0952a983bd62ab8a397763b60475c0`. Validate source data IDs/order and every pinned input hash before running. Stop on a mismatch; do not silently accept newer files or alter arithmetic to fit the hash.
3. Prepare one direct derivative: change only **TAB_W0.08→0.07**, preserve NINA_STEP5 and all other settings/inputs/arithmetic. Motivation is the source's measured trend0.15worse,0.10/0.08stronger; this does not prove further reduction helps. No sweep, label-free winner selection, extrapolation increase or new learned model. Export a clearly labeled unscored candidate CSV and an attributed edited notebook for owner review, alongside unchanged reproduction and configuration/input/output receipts.
4. Report exact changed setting, test-rank displacement, hashes and artifact paths. Rank displacement is descriptive, not an AUC estimate. Exact Summit OOF remains absent. No score improvement can be established without valid matching validation or an authorized later score; no submission here.

## Ownership and verification

Root owns this plan, own strategy entry and public-safe ledger summary; local task-owned artifact folders `research/summit-361447722/composer-inputs/` and `results/summit-composer-adaptation/`. No other runtime module changes are planned. Assert-based numerical reproduction/shape/ID/finiteness/hash checks cover the adaptation. No full software-suite rerun is needed if runtime source remains unchanged.

Status: CLAIMED; public inputs/reproduction pending. The already frozen Summit baseline and historical control/failed pilot evidence remain preserved.
