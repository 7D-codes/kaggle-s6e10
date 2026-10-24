"""Original-only expected service ratings for competition rows."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_teacher import encode_categories, exclude_competition_overlaps, require_cuda


RATING_COLUMNS = (
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
)
NON_RATING_COLUMNS = {"Age", "Flight Distance", "Departure Delay in Minutes",
                      "Arrival Delay in Minutes", "Gender", "Customer Type", "Type of Travel", "Class"}


def run(args, targets=None):
    """Fit one original-only classifier per rating; targets limits work for small tests."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.rounds < 1 or args.device not in ("cpu", "cuda"):
        raise ValueError("rounds must be positive and device must be cpu or cuda")
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    paths = {"original": Path(args.original), "train": Path(args.data) / "train.csv",
             "test": Path(args.data) / "test.csv"}
    original, train, test = (pd.read_csv(paths[name]) for name in ("original", "train", "test"))
    if not train.id.is_unique or not test.id.is_unique or set(train.id) & set(test.id):
        raise ValueError("Competition IDs must be unique across train and test")
    if "satisfaction" not in original or "satisfaction" not in train or "satisfaction" in test:
        raise ValueError("Expected satisfaction only in original and competition train")
    source, removed = exclude_competition_overlaps(original, train, test)
    columns = [c for c in train if c not in ("id", "satisfaction")]
    chosen = list(RATING_COLUMNS if targets is None else targets)
    if (not chosen or len(chosen) != len(set(chosen)) or
            any(c not in RATING_COLUMNS or c not in columns for c in chosen)):
        raise ValueError("Targets must be distinct competition rating columns")
    if targets is None and (set(columns) != set(RATING_COLUMNS) | NON_RATING_COLUMNS or len(columns) != 21):
        raise ValueError("Expected exactly 13 rating and 8 non-rating features")
    if not len(source):
        raise ValueError("No original rows remain after overlap removal")
    scoring = pd.concat([train[columns], test[columns]], ignore_index=True)
    result = pd.DataFrame({"id": pd.concat([train.id, test.id], ignore_index=True)})
    config = dict(n_estimators=args.rounds, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.7, random_state=args.seed,
                  n_jobs=6, tree_method="hist", enable_categorical=True, device=args.device)
    predictor_columns, target_class_values, vocabularies = {}, {}, {}
    for index, target in enumerate(chosen):
        predictors = [c for c in columns if c != target]
        # Never let either satisfaction label or the observed rating enter X.
        fitted, transformed, levels = encode_categories(source[predictors], scoring[predictors])
        labels = LabelEncoder()
        y = labels.fit_transform(source[target])
        if len(labels.classes_) < 2 or not np.issubdtype(labels.classes_.dtype, np.number):
            raise ValueError(f"Rating {target} needs at least two numeric classes")
        if args.device == "cuda":
            # Exercise all classes on the GPU before starting the full 400-tree fit.
            sample = np.concatenate([np.flatnonzero(y == c)[:2] for c in range(len(labels.classes_))])
            smoke = xgb.XGBClassifier(**{**config, "n_estimators": 2})
            smoke.fit(fitted.iloc[sample], y[sample])
            require_cuda(smoke)
        model = xgb.XGBClassifier(**config)
        model.fit(fitted, y)
        if args.device == "cuda":
            require_cuda(model)
        class_values = labels.classes_.astype(float)
        expected = np.clip(model.predict_proba(transformed) @ class_values,
                           class_values.min(), class_values.max())
        if not np.isfinite(expected).all():
            raise ValueError(f"Nonfinite expected rating for {target}")
        result[f"aux_expected_{target}"] = expected
        model.save_model(output / f"aux-model-{index:02d}.json")
        predictor_columns[target] = predictors
        target_class_values[target] = labels.classes_.tolist()
        vocabularies[target] = levels
    provenance = {
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in paths.items()},
        "original_rows": len(original), "original_exact_overlap_removed": removed,
        "original_rows_used": len(source), "competition_train_rows": len(train),
        "competition_test_rows": len(test), "targets": chosen,
        "target_class_values": target_class_values, "predictor_columns": predictor_columns,
        "categorical_levels": vocabularies[chosen[0]], "predictor_vocabularies": vocabularies,
        "configuration": config,
        "versions": {name: version(name) for name in ("xgboost", "pandas", "scikit-learn")},
        "submissions": 0,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    # Keep all model targets/configuration available before publishing features.
    result.to_csv(output / "auxiliary-features.csv", index=False)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing train.csv and test.csv")
    parser.add_argument("--original", required=True, help="Original data.csv")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
