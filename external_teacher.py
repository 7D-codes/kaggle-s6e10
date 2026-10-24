"""Original-only XGBoost teacher for competition train/test features."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
import pandas as pd


def exclude_competition_overlaps(original, train, test):
    """Discard source rows with any full-feature match in either competition split."""
    columns = [c for c in train if c not in ("id", "satisfaction")]
    if set(columns) != set(test) - {"id"} or set(columns) != set(original) - {"satisfaction"}:
        raise ValueError("Original and competition feature columns differ")
    source = pd.MultiIndex.from_frame(original[columns])
    competition = pd.MultiIndex.from_frame(pd.concat([train[columns], test[columns]], ignore_index=True))
    overlap = source.isin(competition)
    return original.loc[~overlap].reset_index(drop=True), int(overlap.sum())


def encode_categories(source, scoring):
    """Learn categorical levels only from the retained source data."""
    fitted, transformed = source.copy(), scoring.copy()
    levels = {}
    for column in source.select_dtypes(include=["object", "string", "category"]):
        levels[column] = source[column].dropna().unique().tolist()
        for frame in (fitted, transformed):
            values = frame[column].where(frame[column].isin(levels[column]))
            frame[column] = pd.Categorical(values, categories=levels[column])
    return fitted, transformed, levels


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.rounds < 1 or args.device not in ("cpu", "cuda"):
        raise ValueError("rounds must be positive and device must be cpu or cuda")
    import xgboost as xgb
    from features import encode_original_target

    paths = {"original": Path(args.original), "train": Path(args.data) / "train.csv",
             "test": Path(args.data) / "test.csv"}
    original, train, test = (pd.read_csv(paths[name]) for name in ("original", "train", "test"))
    if not train.id.is_unique or not test.id.is_unique or set(train.id) & set(test.id):
        raise ValueError("Competition IDs must be unique across train and test")
    if "satisfaction" in test or "satisfaction" not in train or "satisfaction" not in original:
        raise ValueError("Expected satisfaction only in original and competition train")
    source, removed = exclude_competition_overlaps(original, train, test)
    y = encode_original_target(source["satisfaction"])
    if not len(source) or y.nunique() != 2:
        raise ValueError("Retained original data must contain both target classes")
    columns = [c for c in train if c not in ("id", "satisfaction")]
    scoring = pd.concat([train[columns], test[columns]], ignore_index=True)
    fitted, scoring, levels = encode_categories(source[columns], scoring)
    config = dict(n_estimators=args.rounds, learning_rate=0.05, max_depth=8,
                  subsample=0.8, colsample_bytree=0.5, random_state=args.seed,
                  n_jobs=6, tree_method="hist", enable_categorical=True,
                  eval_metric="logloss", device=args.device)
    if args.device == "cuda":
        # Catch silent GPU-to-CPU fallback before the expensive full fit.
        sample = pd.concat([y[y == label].index[:64].to_series() for label in (0, 1)]).to_numpy()
        smoke = xgb.XGBClassifier(**{**config, "n_estimators": 2})
        smoke.fit(fitted.iloc[sample], y.iloc[sample])
        require_cuda(smoke)
    model = xgb.XGBClassifier(**config)
    model.fit(fitted, y)
    if args.device == "cuda":
        require_cuda(model)
    probabilities = model.predict_proba(scoring)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Teacher returned nonfinite probabilities")
    model.save_model(output / "teacher-model.json")
    pd.DataFrame({"id": pd.concat([train.id, test.id], ignore_index=True),
                  "teacher_original_xgb": probabilities}).to_csv(
                      output / "teacher-features.csv", index=False)
    provenance = {"input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for name, path in paths.items()},
                  "original_rows": len(original), "original_exact_overlap_removed": removed,
                  "original_rows_used": len(source), "competition_train_rows": len(train),
                  "competition_test_rows": len(test), "features": columns,
                  "categorical_levels": levels, "configuration": config,
                  "versions": {name: version(name) for name in ("xgboost", "pandas", "scikit-learn")},
                  "submissions": 0}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def require_cuda(model):
    actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not actual.startswith("cuda"):
        raise RuntimeError(f"XGBoost CUDA requested but booster actually ran on {actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing train.csv and test.csv")
    parser.add_argument("--original", required=True, help="Competition-linked original data.csv")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
