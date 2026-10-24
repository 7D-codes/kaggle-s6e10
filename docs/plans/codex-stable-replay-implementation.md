# Canonical TabPFN replay implementation

Owner: Codex stable replay implementation / GPT-6. Branch: `agent/codex/tabpfn-replay-fix`. Explicit root handoff covers the four runner/packager/test files and this plan/own registry entry only.

## Scope and contract

Use exactly 128 native query rows for every production/replay dispatch, splitting in order and repeating the final real row for short tails. Reject empty input and malformed backend probabilities; retain global nonconstant checks without rejecting valid one-row repeated blocks. Preserve FP32, int8 KV/cache/device settings, seed/context/data/checkpoint provenance and original `rtol=1e-6, atol=1e-7` checks.

Add exact native128 anchor, fixed-seed nonprefix sizes1/17/63/65/127/128/129, 1000-row permutation/inversion and duplicate-row slot checks; record comparison evidence before assertions. Capture fitted CPU preprocessing configuration/equality where exposed, with no GPU-equivalence claim. Track actual native calls/shapes and complete1000-row runtime. Set private Stable Pilot identity and 1500-second total deadline; never launch here.

## Verification and progress

- Read AGENTS, strategy, experiments, onboarding and root replay-fix contract. Scoped callers: all four current native prediction calls in runner; packager embeds runner source; targeted tests cover runner and generated private package.
- Initial clean worktree at `74a6d33`; configured Git identity `7D <miuatoro.exe@gmail.com>`. Fetched origin; branch initially has no upstream, so initial pull reports no tracking branch. Latest origin/main read; no coordination change requires integration.
- Baseline: 71 passed, 66 existing warnings, 14.21s using `/tmp/s6e10-test-env/bin/python -m pytest -q`.
- RED: 8 failures demonstrated missing canonical dispatch/receipts/stable package metadata; 5 further failures demonstrated missing native/CPU calibration; separate RED verified missing CPU max-error evidence. Follow-up review RED covered missing explicit cache guard, config hash and direct17 diagnostic flag (4 failures, 2 passed).
- GREEN: targeted runner/packager suite 36 passed in 2.09s. Prior full checkpoint 81 passed, 66 existing warnings, 10.86s; fresh source-only Git archive at code `336d9711ae10f77373f4912a9adfa2998fac6dad`: **84 passed**, 66 existing warnings, 11.89s (`/tmp/s6e10-stable-replay-source-0loYVM`, same Python environment). No ignored data/artifacts or CUDA/model installs were required.
- Implemented one fixed128 native boundary and universal query splitting/padding/order retention; strict binary probability validation allows legitimate constant/single-row blocks while real validation/test/replay64 remain nonconstant. Original repeat64/cross1000_64 gates retain `rtol=1e-6, atol=1e-7` and record each result before asserting. Synthetic per-call/model drift remains rejected.
- Native checks include direct128 real-row anchor, nonprefix sizes1/17/63/65/127/128/129, fixed-seed1000-row permutation/inversion and rows0/1 repeated in all128 slots against the direct mixed anchor. Dispatch count/shapes, synchronized complete1000 validation runtime and active fitted member config are recorded.
- Root prelaunch review clarified direct17 CPU transform is diagnostic only (exact/equal-NaN/max-error/original-tolerance evidence), because arbitrary native query sizes are outside the canonical recipe. Padded17 physical128 and repeated-row physical128 CPU composition checks gate exact equal-NaN equality; native probability gates keep original tolerances. CPU evidence explicitly excludes GPU preprocessing.
- Private stable package ID/title and total setup/download/inference cap1500seconds; all T4/privacy/single-device/data/checkpoint hash guards retained. Config SHA256 covers scientific params/folds/features/categories and dispatch recipe, excluding local checkpoint path and runtime counters; unchanged bytes at a new checkpoint/output path retain the config hash.
- No real model/data download, installation, training, GPU job or submission.

## Handoff

Code `336d9711ae10f77373f4912a9adfa2998fac6dad` was pushed and remote SHA read back exactly. Staged diff/whitespace reviewed; only four authorized runtime/test files and own coordination entries changed. Exactly one native `predict_proba` call remains at the asserted128 boundary. Independent root specification/quality review remains pending; ownership retained for review corrections. Root owns integration and any separately authorized native verification. Native reproducibility remains unverified until that job passes.
