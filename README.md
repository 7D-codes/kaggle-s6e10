# Predicting Airline Satisfaction: Kaggle S6E10

AI-assisted experiments for [Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10), evaluated with **ROC AUC**.

## Verified result

- Best public submission from this project: **0.96111**, submission `57568517`.
- Submitted model: **50% RealMLP / 25% full-recipe LightGBM / 25% full-recipe XGBoost**, fixed arithmetic weights.
- Submitted seed-2026 development OOF ROC AUC: **0.9614641415242471**.
- Matched seed-2026 prior ensemble: **0.9613436210433547**. The improvement was positive in all five paired folds and also repeated on seed 42.

OOF means held-out predictions collected across five validation folds. These are development-exposed rows, not an untouched holdout. OOF scores are not public leaderboard scores, and a higher development score does not guarantee a better submission.

The current research target is public ROC AUC 0.962 or higher. That is an objective, not an achieved result or a guarantee. New training is paused for a diagnostic review rather than unrestricted feature or hyperparameter search.

## Repository contents

- `train_baselines.py`: fold-based LightGBM, XGBoost and CatBoost training.
- `train_realmlp.py`: fixed-epoch PyTabKit RealMLP, with outer validation labels excluded from fitting.
- `features.py`, `recipe_features.py`: source-data priors, fold-local statistics, route/digit/frequency features and cross-fitted target encoding.
- `external_teacher.py`, `external_auxiliary.py`: original-source teacher and auxiliary rating predictors.
- `predict_realmlp.py`, `predict_recipe.py`: raw-row feature reconstruction and checkpoint inference.
- `blend_equal.py`: aligned fixed equal-weight blending, preserving split provenance.
- `verify_results.py`: independent IDs, targets, probability ranges, fold membership and AUC verification.
- `package_*.py`: reproducible private Kaggle job packaging.
- `test_*.py`: feature safety, training controls, inference, generated smoke inputs and blending regression tests.

Datasets, predictions, fitted models, local reports/logs and downloaded third-party research are intentionally excluded. This is a source-code repository, not a downloadable trained-model release.

## Data setup

Install [uv](https://docs.astral.sh/uv/) and authenticate with Kaggle using its official credential setup. Never commit credentials.

```bash
mkdir -p data
uv run --with kaggle kaggle competitions download \
  -c playground-series-s6e10 -p data
unzip data/playground-series-s6e10.zip -d data
```

Accept competition rules on Kaggle if required. Source-data experiments additionally need the competition-linked public dataset [`arseniyshutko/binary-aviation-satisfaction-129k`](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k):

```bash
mkdir -p data/original
uv run --with kaggle kaggle datasets download \
  -d arseniyshutko/binary-aviation-satisfaction-129k -p data/original
unzip data/original/binary-aviation-satisfaction-129k.zip -d data/original
```

Verify the extracted original CSV is at `data/original/data.csv`. Source-only fitting excludes exact feature overlaps with both competition train and test rows. See script `--help` output before running experiments; use fresh output directories to preserve previous artifacts.

## Tests

The last full local regression run passed **48 tests**. Dependencies are installed into an isolated uv environment:

```bash
uv run --with pandas --with scikit-learn --with pytest --with joblib \
  --with lightgbm --with xgboost --with cloudpickle==3.1.2 \
  python -m pytest -q
```

On macOS, LightGBM may require `brew install libomp`. Full RealMLP training requires PyTabKit and a suitable PyTorch runtime; the remote experiment used PyTabKit 1.7.3, PyTorch 2.11.0+cu128, NumPy 2.1.3, pandas 2.3.3 and scikit-learn 1.6.1. Tree and historical neural runs have their own recorded versions; these are not universal pins for every model.

## Accepted and rejected experiments

| Experiment | Outcome |
|---|---|
| Original-data priors and source-only teacher | Improved selected tree controls |
| Route/frequency/digit/cross-fitted encoding recipe | Improved the fixed neural/tree ensemble on two splits |
| Teacher-enhanced neural model | Rejected: no improvement over retained model |
| Source-only auxiliary rating predictors | Rejected: final fixed blend did not improve |
| RealMLP 6 versus 3 fixed epochs | Rejected: fixed blend worsened in all five paired folds |
| Explicit nonzero service-rating count/mean | Tiny development gain, only 3/5 positive folds; failed predeclared promotion gate |

## Reproducibility boundaries

- Preserve split seeds and align IDs, targets and fold assignments before blending. Do not combine OOF arrays from different splits as if their held-out models matched.
- Raw-row reconstruction and saved-prediction verification are separate checks.
- New recipe states freeze fitted rarity lookup values to avoid architecture-dependent float32-log drift. Historical artifacts are not retroactively changed.
- CUDA spot checks passed for the submitted tree recipe. Strict XGBoost CPU replay at `rtol=0, atol=1e-7` failed for 1 of 128 checked rows, with maximum difference `1.043081283569336e-7`. Strict CPU equivalence is **not** claimed.
- Neural raw-row CUDA checkpoint replay passed on 64 nonconstant rows per tested arm at its original `rtol=1e-6, atol=1e-7`. This does not prove full-test or CPU equivalence.
- Never load untrusted pickle/joblib/cloudpickle checkpoints. Arrow-backed preprocessing vocabularies can require PyArrow even without reading a Parquet file.
- Packagers generate job sources from current local modules. Publishing this repository does not publish private Kaggle model outputs, launch a job or authorize a competition submission.

## Research attribution

The project adapts and evaluates public ideas rather than claiming a new algorithm:

- [`busyaprime/s6e10-route-or-distance-both-lb-0-96169`](https://www.kaggle.com/code/busyaprime/s6e10-route-or-distance-both-lb-0-96169): route/digit/frequency and target-encoding recipe.
- [`yekenot/ps-s6-e10-realmlp-pytabkit`](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit): RealMLP/PyTabKit reference.
- [`aryakghosal/airline-passenger-satisfaction-analysis`](https://github.com/aryakghosal/airline-passenger-satisfaction-analysis): descriptive analysis and rating dictionary, where zero means not applicable.

Public notebook titles or their reported scores are not treated as independently reproduced results. Data remains subject to its own terms and competition rules. Implementation and verification were performed with AI assistance.
