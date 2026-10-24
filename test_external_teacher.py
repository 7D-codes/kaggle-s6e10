"""Contract tests for the original-only teacher (tiny synthetic data)."""
from argparse import Namespace
import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from external_teacher import exclude_competition_overlaps, encode_categories, require_cuda, run


def test_overlap_removal_uses_train_and_test_features_not_labels():
    original = pd.DataFrame({"Age": [10, 20, 30, 40], "Gender": ["A", "B", "C", "D"],
                             "satisfaction": [0, 1, 0, 1]})
    train = pd.DataFrame({"id": [1], "Age": [10], "Gender": ["A"], "satisfaction": [1]})
    test = pd.DataFrame({"id": [2], "Age": [20], "Gender": ["B"]})
    kept, removed = exclude_competition_overlaps(original, train, test)
    assert removed == 2
    assert kept.Age.tolist() == [30, 40]


def test_category_vocabulary_is_source_only_and_unseen_becomes_missing():
    source = pd.DataFrame({"Age": [30, 40], "Gender": ["C", "D"]})
    competition = pd.DataFrame({"Age": [50], "Gender": ["COMPETITION_ONLY"]})
    fitted, scoring, levels = encode_categories(source, competition)
    assert levels == {"Gender": ["C", "D"]}
    assert fitted.Gender.cat.categories.tolist() == ["C", "D"]
    assert pd.isna(scoring.Gender.iloc[0])


def test_existing_output_refuses_before_reading_missing_inputs(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    args = Namespace(output=output, data=tmp_path / "nonexistent", original=tmp_path / "missing.csv",
                     rounds=2, device="cpu", seed=42)
    with pytest.raises(FileExistsError):
        run(args)
    assert (output / "sentinel").read_text() == "preserve"


def test_tiny_cpu_fit_writes_ordered_features_model_and_provenance(tmp_path):
    data = tmp_path / "competition"
    data.mkdir()
    train = pd.DataFrame({"id": [8, 3], "Age": [10, 99], "Gender": ["A", "UNSEEN"],
                          "satisfaction": [1, 0]})
    test = pd.DataFrame({"id": [7, 4], "Age": [20, 30], "Gender": ["B", "C"]})
    original = pd.DataFrame({"Age": [10, 20, 30, 31, 32, 33, 34, 35, 36, 37],
                             "Gender": ["A", "B", "C", "C", "C", "D", "D", "D", "C", "D"],
                             "satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied",
                                              "satisfied", "neutral or dissatisfied", "satisfied",
                                              "neutral or dissatisfied", "satisfied",
                                              "neutral or dissatisfied", "neutral or dissatisfied"]})
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    original_path = tmp_path / "data.csv"
    original.to_csv(original_path, index=False)
    output = tmp_path / "result"
    run(Namespace(data=data, original=original_path, output=output, rounds=2, device="cpu", seed=42))
    result = pd.read_csv(output / "teacher-features.csv")
    assert result.columns.tolist() == ["id", "teacher_original_xgb"]
    assert result.id.tolist() == [8, 3, 7, 4]
    assert len(result) == len(train) + len(test)
    assert np.isfinite(result.teacher_original_xgb).all()
    assert result.teacher_original_xgb.between(0, 1).all()
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(output / "teacher-model.json")
    assert model.n_features_in_ == 2
    with pytest.raises(RuntimeError, match="actually ran on cpu"):
        require_cuda(model)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["original_exact_overlap_removed"] == 3
    assert provenance["original_rows_used"] == 7
    assert provenance["categorical_levels"] == {"Gender": ["C", "D"]}
    assert provenance["configuration"]["n_estimators"] == 2
    assert provenance["configuration"]["max_depth"] == 8
    assert provenance["configuration"]["device"] == "cpu"
    assert provenance["submissions"] == 0
    assert provenance["versions"]["xgboost"] == xgb.__version__
    for name, path in {"original": original_path, "train": data / "train.csv",
                       "test": data / "test.csv"}.items():
        assert provenance["input_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
