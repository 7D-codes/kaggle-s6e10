# Codex: bounded TabPFN plan execution

Owner: Codex / gpt-6.1-sol. Branch: `agent/codex/bounded-tabpfn`. Worktree: `.worktrees/codex-tabpfn`. The owner explicitly authorized execution of [the research plan](codex-breakthrough-research.md). Training/GPU pilots in its caps are resumed; competition submissions remain unauthorized.

## Scope and order

1. Recover exact seed-42/2026 incumbent predictions and competition data using authorized Kaggle access; validate public-safe hashes and fold/ID/target alignment.
2. Verify checkpoint access and GPU feasibility. Do not assume an API token grants gated weight access or available GPU quota.
3. Implement only the required fold-local raw-feature TabPFN runner, private Kaggle packager and fixed-candidate comparison with synthetic TDD tests; preserve existing modules and inference tolerances.
4. Freeze the prelaunch contract and publish code/provenance before a capped pilot. Pilot <=30 minutes on one GPU; first full fold <=2 GPU-hours; projected full seed-42 run including test predictions <=10 GPU-hours. Stop and report resource/access failures.
5. Evaluate the fixed `0.8 * incumbent + 0.2 * candidate` blend, full coverage only. Gate: pooled ROC AUC gain >=0.00005, >=4/5 positive paired folds, paired stratified-bootstrap 95% lower endpoint >0. Confirm passing frozen recipe on seed 2026 with the same gate. Reused-row validation remains development evidence.
6. Conditional TabM stage is one configuration, <=5 GPU-hours, after resource/access requirements are resolved. Learned combination requires a separate validation design within the approved research plan; no broad sweep.

## Initial verification and limits

- Main was clean and current with `origin/main` at `a4baaa2b8e0d89bd5e69def7d09693ec3d9b034b`.
- Local environment has no GPU tooling, competition data, incumbent predictions, Kaggle credentials or TabPFN token. The owner-supplied screenshot is available; any authentication uses its credential only in process memory, without printing it.
- Private output downloads are limited to needed CSV/JSON evidence first; do not load foreign pickle/joblib/cloudpickle models.
- No experiment job, GPU smoke or submission launched yet. Runtime/configuration/checkpoint identity remains to be pinned after access checks.

## Progress and handoff

Access checks and isolated software verification in progress. Root owns coordination; implementation ownership will be handed to a separate worktree if delegated. Record exact tests, remote versions, artifact hashes, resource use and final decision here and in the shared ledger.

### Access and exact comparator recovery

- Owner reiterated **“fyi don't submit”**. No competition submission is allowed; this restriction persists through closeout.
- Owner-supplied credential authenticated in process memory. No credential file installed. Kaggle reports 0 of 30 weekly GPU hours used, refresh 2026-10-31 00:00:00 (reported UTC value); quota must be rechecked before launch.
- Competition train/test SHA256 matches historical provenance: train `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36`; test `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6`. Verified 699635 train and 299844 test rows, unique aligned IDs and binary targets.
- Recovered only required model CSV/JSON files using authorized access. Exact seed42 neural source is `wasidyms/s6e10-batch2-realmlp-gpu/2` (version1 has no completed outputs); seed2026 neural source is `wasidyms/s6e10-realmlp-confirm-seed2026/1`. Trees are the Batch6 route recipe/confirmation version1 arms recorded in the ledger.
- Reconstructed fixed 50/25/25 controls after checking every member's IDs, targets, exact five-fold assignments, probability ranges and metric agreement. Independently recomputed AUC: seed42 **0.9614620347107486**; seed2026 **0.9614641415242471**, exact ledger matches.
- Reconstructed seed42 OOF SHA256 `eb4e410a91cf6028681c1164f36bd59aa458b3a5b71c842cd3d8d1840c6bf53b`; test SHA256 `226a1b3cd43839ee3040be8da2c61132f438dff4d7fb9c71d5b937ea103eace5`. Seed2026 OOF SHA256 `403997c37a46c0e0bd5273005014cb595e1d55416ee2a72b74f8d35a533a5252`; test SHA256 `bab9f1cebcb02cf526587786a5620f208fde9d7d34332531293a0c02b011617e`. CSV round trips do not establish byte identity to retained submission57568517; its original recorded hash remains preserved.
- Safe-format full evidence/receipts remain local under `results/recovered/`; no checkpoint deserialization occurred. Do not commit these files.
- Fresh source-only baseline: **48 passed**, 66 existing warnings, 13.89s with Python3.13.11, NumPy2.5.4, pandas3.1.0, sklearn1.9.1, XGBoost3.4.1, LightGBM4.7.0, cloudpickle3.1.2. These are software-test versions, not a claim to reproduce historical training runtimes.
- Delegated runner/packager/tests to Codex subagent `tabpfn_runner` in `.worktrees/codex-tabpfn-code`, branch `agent/codex/tabpfn-runner`. Root retains artifact recovery and this coordination plan. Separate spec and quality reviews required before launch.

### Pilot contract — frozen statistical choices, software verification pending

- Hypothesis: standard supported TabPFN-3.5 inference can produce valid nonconstant probabilities within a T4 resource envelope; this pilot does not test a full OOF improvement.
- Execution owner/monitor: root Codex only. Planned private kernel `wasidyms/s6e10-codex-tabpfn-pilot`, version unknown until push/readback. No duplicate launch.
- Data/split: verified competition files above; `StratifiedKFold(5, shuffle=True, random_state=42)`. Context is first50000indices of fold0's training rows; validation/test are first1000eligible rows each. Label-free category maps fit only on context. Preserve numeric missingness; no source teacher, TE, external data or extra feature arms.
- Model: `tabpfn==9.0.0`, checkpoint `Prior-Labs/tabpfn_3_5/tabpfn-v3.5-20260909.safetensors`, repository revision `b0662010dd17679509f42326e2f9ba37a6f58818`, checkpoint SHA256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3` (876027932bytes), verified against official Hugging Face LFS metadata before download.
- Parameters: one estimator; random_state42; devicecuda:0; fit_modefit_with_cache; int8KVcache; autocast inference; memory_saving_modeTrue; keep_cache_on_deviceFalse. No custom attention patch or unregistered fallbacks. Actual runtime versions/device/memory must be recorded by job.
- Resource cap: hard1800seconds including package setup, checkpoint download and worker inference; worker subprocess group terminated at cap. One visible GPU even if notebook allocation exposes two. No use of second GPU for another arm. Pilot predictions and timing are screening evidence only.
- Acceptance: finite [0,1] nonconstant predictions; exact row IDs and fold membership; no fit sees outer validation labels; repeated inference passes predeclared `rtol=1e-6, atol=1e-7`; checksum/runtime/source receipts complete. Stop on timeout, OOM, version/config mismatch, incomplete output or failed replay. Do not relax tolerance or silently shrink context.
- Full-fold promotion remains conditional on pilot acceptance and a separately recorded <=2GPU-hour first-full-fold contract, followed by a forecast respecting <=10GPU-hour full seed42cap. No full-run quality decision from the 1000-row pilot.
