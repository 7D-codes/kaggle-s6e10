# Summit Composer audit and handoff

Owner: root Codex / gpt-6.1-sol. Branch: `agent/codex/bounded-tabpfn`. This is steering within [the active bounded execution](codex-bounded-tabpfn.md). The owner requested building on this notebook and explicitly prohibited competition submissions. No submission is authorized.

## Exact source and findings

The supplied [scriptVersionId361447722](https://www.kaggle.com/code/mayankraaz/s6e10-summit-composer-0-96210?scriptVersionId=361447722) was downloaded directly through Kaggle's public script-content endpoint. Its cell sources match API version2. Despite the URL title, this version describes a **0.96213** public result. It blends inherited public files with TabPFN, selected with public feedback. Its text reports that analogous foundation-model additions hurt its 181-member stack's OOF. The exact Summit output has no accompanying OOF in its output listing.

The linked [public atlas](https://www.kaggle.com/datasets/mayankraaz/s6e10-lb-surface-atlas) contains a separate 181-member OOF/test pair and metadata. Its metadata lists five-fold and ten-fold members but provides neither training IDs nor a per-row fold mapping. Its stack OOF must not be silently equated with the exact Summit output or merged with our folds.

Source inspection also found that the `peel` transformation estimates hidden magnitudes using a reference distribution. It does not identify the original hidden vector uniquely from ranks. We will use public outputs only; no private source or private output access is required or attempted.

## Independent verification

- Exact notebook download SHA256: `2e6fd8d04ea9e6e7f43dca822fac115382d498348aa6e89bd10ac9b24658dfbc`.
- Normalized cell-source SHA256: `9626cd010571797f878e6f300262902bec694def7d20dae7d51a5e2f943cc609`; API `blob.source` SHA256: `5938ed7bc00e3ec2dc79e90c19c08bd2f6f0a17f420cc1961253b7819e26c366`. Notebook JSON representations differ, cell sources match.
- Downloaded public version2 CSV only. Checked exact competition test IDs/order, uniqueness, 299844rows and finite [0,1] values. CSV SHA256: `0c6d65f427725bd57ed2885826c451f6eb0952a983bd62ab8a397763b60475c0`.
- CSV export rounds to nine decimals. Re-ranking its parsed values reconstructs the notebook's declared internal rank vector hash exactly: `21f6fe37dd8f1b7673677d00d6d1bbd75190534d0d557dee76b9fd897a4fdfed`. This establishes the public output's ranking identity, not its actual unseen-label score.
- Loaded public NPY files with `allow_pickle=False`; checked shape, finiteness and [0,1]. Atlas OOF SHA256: `3103c144d2a7ae97c284594cfda6da870e5bb13f509a4333de711f7b44e66088`; test SHA256: `1f2fc66c782a8e1175db33318b272d1194b71d22a98fbbbfb54a1814a1b31b39`, matching the notebook's declared test hash.
- Assuming competition row order, independently recomputed atlas OOF ROC AUC **0.9621655407199953**, matching its metadata. This equality supports row-order consistency but does not replace a member/fold/provenance audit or establish pristine/nested validation.
- Test rank correlation: atlas vs Summit **0.9949542332499519**; our seed42 incumbent vs Summit **0.9850578595750916**. These are label-free comparisons and do not select a better prediction vector.
- Tiny synthetic assert-based check constructed two distinct hidden base vectors yielding identical observed blended ranks. The same reconstructed surrogate re-blended to the observed ranks while differing from the true second vector. Thus round-trip rank equality cannot prove exact hidden-base recovery.

Raw notebook/public predictions and private receipts remain ignored under local `research/summit-361447722/` or `/tmp/s6e10-summit`; no copied third-party notebook or arrays are committed. No foreign checkpoint was loaded.

## How this changes execution

1. Preserve the exact public ranking as an external comparison anchor; preserve the project's own retained incumbent independently.
2. Requested the friend's exact Summit OOF with training IDs/folds through an optional user question. Pending; its absence does not block independently testing the already authorized model hypothesis.
3. Continue the reviewed bounded TabPFN pilot. A later complete new member must first demonstrate incremental value on an auditable OOF control. Do not tune a Summit blend from test-only rank correlations or invent its offline AUC.
4. If exact Summit OOF is available, verify its IDs/folds, recipe correspondence and base-model isolation before testing a fixed additive blend. If unavailable, clearly distinguish any later experimental Summit blend from an offline-validated improvement.

No score above the friend's result has been demonstrated. No Summit-derived weights were optimized, no hidden labels used, and no competition submission occurred.
