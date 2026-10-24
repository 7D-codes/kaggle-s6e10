# Canonical TabPFN replay implementation

Owner: Codex stable replay implementation / GPT-6. Branch: `agent/codex/tabpfn-replay-fix`. Explicit root handoff covers the four runner/packager/test files and this plan/own registry entry only.

## Scope and contract

Use exactly 128 native query rows for every production/replay dispatch, splitting in order and repeating the final real row for short tails. Reject empty input and malformed backend probabilities; retain global nonconstant checks without rejecting valid one-row repeated blocks. Preserve FP32, int8 KV/cache/device settings, seed/context/data/checkpoint provenance and original `rtol=1e-6, atol=1e-7` checks.

Add exact native128 anchor, fixed-seed nonprefix sizes1/17/63/65/127/128/129, 1000-row permutation/inversion and duplicate-row slot checks; record comparison evidence before assertions. Capture fitted CPU preprocessing configuration/equality where exposed, with no GPU-equivalence claim. Track actual native calls/shapes and complete1000-row runtime. Set private Stable Pilot identity and 1500-second total deadline; never launch here.

## Verification and progress

- Read AGENTS, strategy, experiments, onboarding and root replay-fix contract. Scoped callers: all four current native prediction calls in runner; packager embeds runner source; targeted tests cover runner and generated private package.
- Initial clean worktree at `74a6d33`; configured Git identity `7D <miuatoro.exe@gmail.com>`. Fetched origin; branch initially has no upstream, so initial pull reports no tracking branch. Latest origin/main read; no coordination change requires integration.
- Baseline suite pending. No real model/download/job/submission.

## Handoff

Root owns independent specification and quality review, integration and any separately authorized native verification. Native reproducibility remains unverified until that job passes.
