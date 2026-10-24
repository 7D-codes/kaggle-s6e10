# Improve the supplied Summit result

Owner: root Codex. Branch: `agent/codex/bounded-tabpfn`. Task: correct the working benchmark after the owner pointed out that the public-code result already exceeds0.96200. No new training/submission is launched by this baseline alignment.

## Working baseline

Use the exact supplied Summit Composer output as the baseline for candidate test predictions: scriptVersion361447722 / API version2. Source describes public ROC AUC0.96213, despite the0.96210URL title. The owner confirms their public-code result is already above0.96200. Its exact submitted score/receipt was not supplied; no independent leaderboard rescore is claimed. Historical own-model0.96111 remains archived as a secondary reproducible control, not the main target to beat.

Public CSV:299844rows, SHA256 `0c6d65f427725bd57ed2885826c451f6eb0952a983bd62ab8a397763b60475c0`; re-ranked vector hash `21f6fe37dd8f1b7673677d00d6d1bbd75190534d0d557dee76b9fd897a4fdfed`. Existing source/provenance findings are in [the pinned audit](codex-summit-audit.md).

## Corrected approach

1. Freeze the exact public CSV and version/hash manifest locally under `results/summit-baseline/`; verify all IDs/order and rank identity against competition test. Preserve those exact baseline bytes for every comparison.
2. Work from Summit's component/source lineage when choosing a change. Its181-member related pool already includes TabPFN, TabM, route features and multiple neural/tree families. A raw50k/one-estimator TabPFN feasibility run is infrastructure evidence, not a demonstrated missing source of signal. Do not treat family novelty or a lower standalone model score as an improvement to this blend.
3. Prefer one reproducible component change with complete OOF and test predictions; reproduce that component's unchanged control under the same folds before judging the change. The public TabPFN layer's interaction encodings and feature selection are a concrete feature-view lead, but must be fitted within folds. It is not a predeclared training contract yet.
4. Use matching complete OOF for claims of incremental gain. The atlas OOF is a different181-member stack with unresolved fold provenance; the exact Summit CSV has no matching OOF. Do not use the old0.96111 control or atlas gain as proof of beating Summit. If exact OOF cannot be recovered, a Summit-plus-new-member test file must be labeled experimental/unvalidated; a fixed blend alone does not establish improvement. Rank correlation is diagnostic, not a selection metric.
5. Preserve the owner's no-submission instruction. No public-score gain can be confirmed from test predictions alone. Native inference acceptance from the completed Stable Pilot is reused as software evidence, without repeating pilots or launching another job under the consumed one-pilot permission.

## Progress and handoff

Existing exact notebook/CSV and public atlas have been re-inspected. The baseline distinction and source-lineage constraint are now explicit. Freeze/readback verification pending. Next model work must target contribution beyond this supplied Summit result, with a separate bounded experiment contract if training is resumed. No improved candidate or score is claimed by this correction.
