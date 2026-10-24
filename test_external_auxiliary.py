"""Original-only auxiliary rating features: small real XGBoost CPU fits."""
from argparse import Namespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from external_auxiliary import NON_RATING_COLUMNS, RATING_COLUMNS, run


def test_default_rating_columns_are_exactly_thirteen():
    assert len(RATING_COLUMNS) == 13
    assert "Departure Delay in Minutes" not in RATING_COLUMNS
    assert "satisfaction" not in RATING_COLUMNS


def test_existing_output_refuses_before_input_reads(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel").write_text("unchanged")
    with pytest.raises(FileExistsError):
        run(Namespace(output=output, data=tmp_path / "missing", original=tmp_path / "missing.csv",
                      rounds=2, device="cpu", seed=42))
    assert (output / "sentinel").read_text() == "unchanged"


def test_original_only_fit_expected_values_and_artifacts(tmp_path):
    data = tmp_path / "competition"
    data.mkdir()
    # One source row overlaps train, another overlaps test (irrespective of labels).
    original = pd.DataFrame({
        "Age": list(range(30, 54)),
        "Gender": ["A", "B"] + ["SOURCE", "OTHER"] * 11,
        "Flight Distance": list(range(100, 124)),
        "Inflight wifi service": [0, 5] + [1, 3, 5, 1, 3, 5, 1, 3, 5, 1, 3] * 2,
        "Food and drink": [5, 0] + [0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2] * 2,
        "satisfaction": ["neutral or dissatisfied", "satisfied"] * 12,
    })
    columns = [c for c in original if c != "satisfaction"]
    train = pd.concat([original.loc[[0], columns], pd.DataFrame([{
        "Age": 90, "Gender": "COMPETITION_ONLY", "Flight Distance": 500,
        "Inflight wifi service": 2, "Food and drink": 1}])], ignore_index=True)
    train.insert(0, "id", [8, 3])
    train["satisfaction"] = ["satisfied", "neutral or dissatisfied"]
    test = pd.concat([original.loc[[1], columns], pd.DataFrame([{
        "Age": 91, "Gender": "COMPETITION_ONLY", "Flight Distance": 501,
        "Inflight wifi service": 4, "Food and drink": 3}])], ignore_index=True)
    test.insert(0, "id", [7, 4])
    original_path = tmp_path / "original.csv"
    original.to_csv(original_path, index=False)
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)

    output = tmp_path / "output"
    run(Namespace(output=output, data=data, original=original_path, rounds=3,
                  device="cpu", seed=42), targets=["Inflight wifi service", "Food and drink"])
    feature = pd.read_csv(output / "auxiliary-features.csv")
    assert feature.columns.tolist() == ["id", "aux_expected_Inflight wifi service",
                                        "aux_expected_Food and drink"]
    assert feature.id.tolist() == [8, 3, 7, 4]
    assert not any("satisfaction" in c or "prob" in c for c in feature.columns)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["original_exact_overlap_removed"] == 2
    assert provenance["original_rows_used"] == 22
    assert provenance["target_class_values"] == {"Inflight wifi service": [1, 3, 5],
                                                   "Food and drink": [0, 2, 4]}
    assert provenance["configuration"]["n_estimators"] == 3
    assert provenance["configuration"]["max_depth"] == 6
    assert provenance["configuration"]["device"] == "cpu"
    assert provenance["categorical_levels"] == {"Gender": ["SOURCE", "OTHER"]}
    assert provenance["predictor_columns"]["Inflight wifi service"] == [
        "Age", "Gender", "Flight Distance", "Food and drink"]
    assert provenance["predictor_columns"]["Food and drink"] == [
        "Age", "Gender", "Flight Distance", "Inflight wifi service"]
    for name, path in {"original": original_path, "train": data / "train.csv",
                       "test": data / "test.csv"}.items():
        assert provenance["input_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()

    import xgboost as xgb
    from external_teacher import encode_categories
    scoring = pd.concat([train[columns], test[columns]], ignore_index=True)
    retained = original.iloc[2:].reset_index(drop=True)
    for index, target in enumerate(("Inflight wifi service", "Food and drink")):
        model = xgb.XGBClassifier()
        model.load_model(output / f"aux-model-{index:02d}.json")
        predictors = provenance["predictor_columns"][target]
        assert model.get_booster().feature_names == predictors
        assert target not in predictors and "satisfaction" not in predictors
        _, encoded, _ = encode_categories(retained[predictors], scoring[predictors])
        values = np.asarray(provenance["target_class_values"][target])
        expected = model.predict_proba(encoded) @ values
        np.testing.assert_allclose(feature[f"aux_expected_{target}"], expected, rtol=1e-6, atol=1e-6)
        assert np.isfinite(expected).all()
        assert feature[f"aux_expected_{target}"].between(values.min(), values.max()).all()
        assert feature[f"aux_expected_{target}"].between(0, 5).all()


def test_default_trains_all_thirteen_on_twenty_other_features(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    original = pd.DataFrame({
        "Age": list(range(30, 54)), "Flight Distance": list(range(100, 124)),
        "Departure Delay in Minutes": [0, 4] * 12,
        "Arrival Delay in Minutes": [1, 5] * 12,
        "Gender": ["Female", "Male"] * 12,
        "Customer Type": ["Loyal", "Disloyal"] * 12,
        "Type of Travel": ["Business", "Personal"] * 12,
        "Class": ["Eco", "Business"] * 12,
    })
    for index, rating in enumerate(RATING_COLUMNS):
        original[rating] = [(row + index) % 6 for row in range(24)]
    original["satisfaction"] = ["satisfied", "neutral or dissatisfied"] * 12
    columns = [*NON_RATING_COLUMNS, *RATING_COLUMNS]
    # Input feature order is authoritative; all thirteen outputs follow that order.
    train = original.loc[[0, 22], columns].copy()
    train.insert(0, "id", [11, 5])
    train["satisfaction"] = ["neutral or dissatisfied", "satisfied"]
    test = original.loc[[1, 23], columns].copy()
    test.insert(0, "id", [17, 9])
    original_path = tmp_path / "original.csv"
    original.to_csv(original_path, index=False)
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    output = tmp_path / "output"
    run(Namespace(output=output, data=data, original=original_path, rounds=2,
                  device="cpu", seed=42))
    feature = pd.read_csv(output / "auxiliary-features.csv")
    assert feature.columns.tolist() == ["id"] + [f"aux_expected_{c}" for c in RATING_COLUMNS]
    assert feature.id.tolist() == [11, 5, 17, 9]
    assert feature.iloc[:, 1:].apply(lambda col: col.between(0, 5).all()).all()
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["targets"] == list(RATING_COLUMNS)
    assert provenance["original_exact_overlap_removed"] == 4
    assert len(list(output.glob("aux-model-*.json"))) == 13
    for target in RATING_COLUMNS:
        assert len(provenance["predictor_columns"][target]) == 20
        assert target not in provenance["predictor_columns"][target]
        assert "satisfaction" not in provenance["predictor_columns"][target]
