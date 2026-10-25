# Trained models and restoration

Download all ZIP assets from this handoff's GitHub Release and verify their SHA-256 values against `SHA256SUMS.txt`. Extract the ZIPs into one clean workspace root. Paths inside each archive preserve the original workspace-relative layout (`artifacts/...`, `data/original/teacher.joblib`, etc.). Extract `source_and_evidence.zip` into the same root.

Before loading serialized models, restore the exact top-level Python modules from `artifacts/code_snapshots/<source-sha>/` as listed under `training_code` in the submitted provenance manifest. Copy those matching snapshot files into the clean workspace root, alongside the archived reference modules. The shared repository has its own same-named modules from a different campaign; do not substitute them or overwrite that repository to restore this handoff. Verify the source hashes before loading.

`model_manifest.json` lists every packaged model/encoder, exact byte count, SHA-256, original relative path and containing ZIP. The winning submission's original evidence includes a superset of models, including a removed LightGBM leg and fold/teacher dependencies. Inclusion is provenance preservation, not a claim that every packaged model contributes to the submitted CSV. Alternative reference models are retained as completed negative experiments.

The source archive contains exact hashed code snapshots, training records, replay audits, fixed reviews, environment lock files and the submitted candidate's provenance. It does not contain credentials, browser session data, copied public submissions, competition CSV datasets or the vendor foundation checkpoint. Obtain authorized input datasets and the official licensed TabPFN checkpoint separately when required. Its recorded input hash is retained in the evidence.

For the numeric V5 full model, the relevant files are `artifacts/reference_catboost/full_refit/model.cbm` and `artifacts/reference_tabpfn_retrain/prepared/full/features.joblib`. The encoder is fitted on full training data. For native V6, convert the encoder output using `reference_catboost_native_policy.native_matrix` before prediction. XGBoost's saved UBJ model uses the numeric encoder output. TabPFN fitted states require the compatible SDK/checkpoint and the saved context/encoder.

These are the original Python serialized objects from a Windows workspace. Match the environment lock files and source module versions. Some objects/code may retain original absolute paths; adapt restoration paths before use. This export verifies checksums and archive integrity; it does not claim that every pipeline has been made portable to a fresh operating system. The saved-model replay evidence documents what was actually tested during the campaign. Load pickle/joblib/cloudpickle artifacts only when you trust their source.

GitHub supports release assets under 2 GiB each; these archives are grouped to meet that limit. Keep the model ZIPs in Release assets rather than committing them to normal Git history. See [GitHub release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

Published model release: https://github.com/7D-codes/kaggle-s6e10/releases/tag/airline-self-trained-20261025

[Publication receipt and GitHub server digests](transfer_receipt.json) verify all 20 uploaded assets. The immutable build inventories were generated before publication; their preparation status describes that earlier snapshot, not the later transfer.
