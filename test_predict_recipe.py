"""Native CPU round-trip tests for sealed recipe inference bundles."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from external_teacher import run as train_teacher
from test_recipe_features import rows
from train_baselines import run as train_baselines
from predict_recipe import _teacher_probabilities, predict_recipe, seal_recipe_bundle


def make_data(root):
    root.mkdir()
    train = rows(110)
    test = rows(17)
    # Distinguish source rows from every competition row without relying on IDs.
    original = rows(140)
    original["Age"] += 5000
    train.insert(0, "id", np.arange(10000, 10110))
    test.insert(0, "id", np.arange(20000, 20017))
    train["satisfaction"] = np.arange(len(train)) % 2
    original["satisfaction"] = np.arange(len(original)) % 2
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    original.to_csv(root / "original.csv", index=False)
    test[["id"]].assign(satisfaction=0).to_csv(root / "sample_submission.csv", index=False)
    return train, test


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("recipe-inference")
    train, test = make_data(root / "data")
    teacher = root / "original-teacher"
    train_teacher(Namespace(output=str(teacher), rounds=3, device="cpu", seed=42,
                            original=str(root / "data/original.csv"), data=str(root / "data")))
    run_dir = root / "recipe-full"
    train_baselines(Namespace(data=str(root / "data"), output=str(run_dir),
                              original=str(root / "data/original.csv"),
                              teacher_csv=str(teacher / "teacher-features.csv"),
                              auxiliary_csv=None, features="recipe_full", models="lightgbm",
                              seed=42, preset="route", rounds=3, leaves=7, gpu=False))
    seal_recipe_bundle(run_dir)
    return root, train, test, teacher, run_dir


def test_sealed_teacher_bundle_reproduces_saved_native_test_predictions_and_new_rows(trained, tmp_path):
    _, _, test, _, run_dir = trained
    with pytest.raises((FileNotFoundError, ValueError), match="manifest|seal"):
        predict_recipe(test.drop(columns="id"), tmp_path / "unsealed")
    with pytest.raises(FileExistsError):
        seal_recipe_bundle(run_dir)
    scores = predict_recipe(test.drop(columns="id"), run_dir)
    expected = pd.read_csv(run_dir / "lightgbm-predictions.csv").satisfaction.to_numpy()
    assert np.ptp(expected) > 1e-6, "Round trip must exercise nonconstant native trees"
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-7)
    novel = test.drop(columns="id").iloc[:3].copy()
    novel.loc[novel.index[0], "Class"] = "never-before-seen"
    novel.loc[novel.index[1], "Age"] = 99999
    assert np.isfinite(predict_recipe(novel, run_dir)).all()


def test_frozen_teacher_recreates_saved_test_probabilities_without_ids(trained):
    _, train, test, teacher, _ = trained
    saved = pd.read_csv(teacher / "teacher-features.csv")
    np.testing.assert_allclose(
        _teacher_probabilities(test.drop(columns="id"), teacher),
        saved.teacher_original_xgb.to_numpy()[len(train):], rtol=0, atol=1e-7)


@pytest.mark.parametrize("feature_kind", ["recipe_te_only", "recipe_full"])
def test_xgboost_recipe_bundle_matches_native_ensemble(trained, feature_kind):
    root, _, test, teacher, _ = trained
    out = root / ("recipe-xgb-" + feature_kind)
    train_baselines(Namespace(data=str(root / "data"), output=str(out),
                              original=str(root / "data/original.csv"),
                              teacher_csv=str(teacher / "teacher-features.csv"),
                              auxiliary_csv=None, features=feature_kind, models="xgboost",
                              seed=42, preset="baseline", rounds=3, leaves=7, gpu=False))
    seal_recipe_bundle(out, model="xgboost")
    expected = pd.read_csv(out / "xgboost-predictions.csv").satisfaction.to_numpy()
    assert np.ptp(expected) > 1e-6, "Round trip must exercise nonconstant native trees"
    np.testing.assert_allclose(predict_recipe(test.drop(columns="id"), out, model="xgboost"),
                               expected, rtol=0, atol=1e-7)


def test_teacher_csv_mismatch_prevents_sealing(trained, tmp_path):
    _, _, _, teacher, run_dir = trained
    replacement = tmp_path / "original-teacher"
    replacement.mkdir()
    for name in ("teacher-model.json", "teacher-features.csv", "provenance.json"):
        (replacement / name).write_bytes((teacher / name).read_bytes())
    (replacement / "teacher-features.csv").write_text("id,teacher_original_xgb\n")
    unsealed = tmp_path / "recipe-full"
    unsealed.mkdir()
    for path in run_dir.iterdir():
        if path.is_file() and path.name != "inference-manifest.json":
            (unsealed / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Teacher|SHA|hash"):
        seal_recipe_bundle(unsealed, replacement)


def test_missing_teacher_is_rejected_for_required_bundle(trained, tmp_path):
    _, _, test, _, run_dir = trained
    with pytest.raises((FileNotFoundError, ValueError), match="teacher"):
        predict_recipe(test.drop(columns="id"), run_dir, tmp_path / "missing")


@pytest.mark.parametrize("artifact", ["teacher-model.json", "provenance.json"])
def test_teacher_artifact_mutation_rejected_after_seal(trained, tmp_path, artifact):
    _, _, test, teacher, run_dir = trained
    replacement = tmp_path / "original-teacher"
    replacement.mkdir()
    for name in ("teacher-model.json", "teacher-features.csv", "provenance.json"):
        (replacement / name).write_bytes((teacher / name).read_bytes())
    (replacement / artifact).write_bytes((replacement / artifact).read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|SHA|integrity"):
        predict_recipe(test.drop(columns="id"), run_dir, replacement)


def test_fold_model_and_metrics_mutation_rejected_after_seal(trained, tmp_path):
    _, _, test, _, run_dir = trained
    replacement = tmp_path / "recipe-full"
    replacement.mkdir()
    for path in run_dir.iterdir():
        if path.is_file():
            (replacement / path.name).write_bytes(path.read_bytes())
    for name in ("lightgbm-fold0.txt", "metrics.json", "lightgbm-preprocessor-fold0.joblib"):
        changed = replacement / name
        old = changed.read_bytes()
        changed.write_bytes(old + b" ")
        with pytest.raises(ValueError, match="hash|SHA|integrity"):
            predict_recipe(test.drop(columns="id"), replacement)
        changed.write_bytes(old)


def test_teacher_free_recipe_te_only_matches_native_predictions(tmp_path):
    train, test = make_data(tmp_path / "data")
    out = tmp_path / "recipe-te"
    train_baselines(Namespace(data=str(tmp_path / "data"), output=str(out),
                              original=str(tmp_path / "data/original.csv"),
                              teacher_csv=None, auxiliary_csv=None, features="recipe_te_only",
                              models="lightgbm", seed=42, preset="route", rounds=3,
                              leaves=7, gpu=False))
    seal_recipe_bundle(out)
    expected = pd.read_csv(out / "lightgbm-predictions.csv").satisfaction.to_numpy()
    assert np.ptp(expected) > 1e-6, "Round trip must exercise nonconstant native trees"
    np.testing.assert_allclose(predict_recipe(test.drop(columns="id"), out), expected,
                               rtol=0, atol=1e-7)
    assert not json.loads((out / "inference-manifest.json").read_text()).get("teacher_required")


def test_explicit_cpu_device_replays_native_teacher_and_ensemble(trained):
    _, _, test, teacher, run_dir = trained
    raw = test.drop(columns="id")
    np.testing.assert_array_equal(_teacher_probabilities(raw, teacher, device="cpu"),
                                  _teacher_probabilities(raw, teacher))
    np.testing.assert_array_equal(predict_recipe(raw, run_dir, device="cpu"),
                                  predict_recipe(raw, run_dir))


@pytest.mark.parametrize("device", ["gpu", "auto", "cuda:7"])
def test_unknown_inference_device_is_rejected_before_reading_artifacts(device, tmp_path):
    with pytest.raises(ValueError, match="device"):
        predict_recipe(pd.DataFrame(), tmp_path, device=device)
    with pytest.raises(ValueError, match="device"):
        _teacher_probabilities(pd.DataFrame(), tmp_path, device=device)
