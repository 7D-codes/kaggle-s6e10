"""Inference from explicitly sealed five-fold recipe tree artifacts.

A seal binds locally preserved files, not their upstream authenticity; verify the
immutable source artifact version before sealing. Never unpickle an untrusted bundle.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from recipe_features import BASE, CATS, transform_recipe


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _paths(run_dir, teacher_dir, model):
    if model not in ("lightgbm", "xgboost"):
        raise ValueError("Only native LightGBM and XGBoost recipe models are supported")
    run_dir = Path(run_dir)
    teacher_dir = Path(teacher_dir) if teacher_dir is not None else run_dir.parent / "original-teacher"
    suffix = "txt" if model == "lightgbm" else "json"
    return run_dir, teacher_dir, ["metrics.json"] + [
        name for fold in range(5) for name in
        (f"{model}-fold{fold}.{suffix}", f"{model}-preprocessor-fold{fold}.joblib")]


def _metadata(run_dir, model):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    provenance = metrics["provenance"]
    entry = metrics[model]
    kind = provenance["arguments"]["features"]
    if (kind not in ("recipe_full", "recipe_te_only") or entry["feature_kind"] != kind
            or provenance["folds"] != 5):
        raise ValueError("Unsupported or inconsistent recipe feature mode")
    if provenance["arguments"].get("auxiliary_csv") is not None:
        raise ValueError("Auxiliary features are not supported for raw-row inference")
    features = provenance["features"]
    if not isinstance(features, list) or len(features) != entry["feature_count"] or len(set(features)) != len(features):
        raise ValueError("Invalid feature order in training provenance")
    teacher = provenance["arguments"].get("teacher_csv") is not None
    if teacher != ("teacher_original_xgb" in features) or teacher != ("teacher_csv_sha256" in provenance):
        raise ValueError("Inconsistent teacher requirement in training provenance")
    return provenance, kind, teacher, features


def _verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Artifact integrity/hash mismatch: {path}")


def seal_recipe_bundle(run_dir, teacher_dir=None, model="lightgbm"):
    """Bind already-verified local training artifacts; never seal on first prediction."""
    run_dir, teacher_dir, names = _paths(run_dir, teacher_dir, model)
    manifest_path = run_dir / "inference-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already sealed: {manifest_path}")
    provenance, kind, required, features = _metadata(run_dir, model)
    teacher_names = ["teacher-model.json", "provenance.json", "teacher-features.csv"] if required else []
    if required and _digest(teacher_dir / "teacher-features.csv") != provenance["teacher_csv_sha256"]:
        raise ValueError("Teacher CSV SHA mismatch with authoritative training metrics")
    if required:
        source = json.loads((teacher_dir / "provenance.json").read_text())
        if (set(source.get("features", [])) != set(BASE) or
                len(source["features"]) != len(BASE) or
                set(source.get("categorical_levels", {})) != set(CATS)):
            raise ValueError("Invalid teacher provenance")
    # Validate every fold before writing the seal. Joblib is trusted only after
    # the caller verified the source version; the seal itself does not attest that.
    for fold in range(5):
        state = joblib.load(run_dir / f"{model}-preprocessor-fold{fold}.joblib")
        expected = features[:-1] if required else features
        if state["feature_order"] != expected or state["mode"] != kind.removeprefix("recipe_"):
            raise ValueError(f"Fold {fold} feature order/mode disagrees with metrics")
    manifest = {"format": 1, "model": model, "feature_kind": kind,
                "features": features, "teacher_required": required,
                "run_hashes": {name: _digest(run_dir / name) for name in names},
                "teacher_hashes": {name: _digest(teacher_dir / name) for name in teacher_names}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _native_xgb_probabilities(frame, path, device):
    import xgboost as xgb

    model = xgb.XGBClassifier(device=device, enable_categorical=True)
    model.load_model(path)
    model.set_params(device=device)
    if model.get_booster().feature_names != list(frame):
        raise ValueError("Native model feature order disagrees with inference")
    scores = model.predict_proba(frame)[:, 1]
    if device == "cuda":
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError(f"Requested CUDA inference fell back to {actual}")
    return scores


def _teacher_probabilities(raw, teacher_dir, device="cpu"):
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    provenance = json.loads((teacher_dir / "provenance.json").read_text())
    columns = provenance["features"]
    if set(columns) != set(BASE) or len(columns) != len(BASE):
        raise ValueError("Unexpected teacher raw feature schema")
    frame = raw[columns].copy()
    for column, levels in provenance["categorical_levels"].items():
        if column not in columns or not isinstance(levels, list):
            raise ValueError("Invalid teacher categorical provenance")
        frame[column] = pd.Categorical(frame[column].where(frame[column].isin(levels)), categories=levels)
    return _native_xgb_probabilities(frame, teacher_dir / "teacher-model.json", device)


def predict_recipe(raw, run_dir, teacher_dir=None, model="lightgbm", device="cpu"):
    """Predict raw rows; device controls XGBoost inference, including the teacher.

    LightGBM stays on CPU. Legacy states recompute rarity logs and may differ
    across numeric runtimes; CUDA alone does not fix that preprocessing drift.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    run_dir, teacher_dir, names = _paths(run_dir, teacher_dir, model)
    manifest_path = run_dir / "inference-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Explicit seal/manifest required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != 1 or manifest.get("model") != model or
            set(manifest.get("run_hashes", {})) != set(names)):
        raise ValueError("Invalid inference manifest")
    _verify_hashes(run_dir, manifest["run_hashes"])
    provenance, kind, required, features = _metadata(run_dir, model)
    teacher_names = {"teacher-model.json", "provenance.json", "teacher-features.csv"} if required else set()
    if (manifest.get("teacher_required") != required or
            set(manifest.get("teacher_hashes", {})) != teacher_names or
            manifest.get("feature_kind") != kind or manifest.get("features") != features):
        raise ValueError("Manifest and training provenance disagree")
    _verify_hashes(teacher_dir, manifest["teacher_hashes"])
    if required and manifest["teacher_hashes"]["teacher-features.csv"] != provenance["teacher_csv_sha256"]:
        raise ValueError("Teacher CSV hash disagrees with training metrics")
    if not isinstance(raw, pd.DataFrame) or set(raw) not in (set(BASE), set(BASE) | {"id"}):
        raise ValueError("Expected DataFrame with exactly the 21 raw features (optional id)")
    raw = raw[BASE]
    teacher_scores = _teacher_probabilities(raw, teacher_dir, device=device) if required else None
    predictions = np.zeros(len(raw), dtype=float)
    for fold in range(5):
        state = joblib.load(run_dir / f"{model}-preprocessor-fold{fold}.joblib")
        expected = features[:-1] if required else features
        if state["feature_order"] != expected or state["mode"] != kind.removeprefix("recipe_"):
            raise ValueError(f"Fold {fold} feature order/mode disagrees with metrics")
        frame = transform_recipe(raw, state)
        if required:
            frame["teacher_original_xgb"] = teacher_scores
        if list(frame) != features:
            raise ValueError("Inference feature order disagrees with training")
        if model == "lightgbm":
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(run_dir / f"{model}-fold{fold}.txt"))
            # LightGBM native serialization replaces spaces in feature names.
            if booster.feature_name() != [name.replace(" ", "_") for name in features]:
                raise ValueError("Native model feature order disagrees with training")
            score = booster.predict(frame)
        else:
            score = _native_xgb_probabilities(frame, run_dir / f"{model}-fold{fold}.json", device)
        predictions += np.asarray(score) / 5
    return predictions
