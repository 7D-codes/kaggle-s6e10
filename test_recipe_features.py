import pickle

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from recipe_features import BASE, CATS, NUMS, PAIRS, prepare_recipe, transform_recipe


def rows(n=70):
    d = {c: np.arange(n) % 5 for c in BASE}
    for c in CATS:
        d[c] = np.where(np.arange(n) % 3 == 0, "first", "second")
    d["Age"] = np.arange(n) + 18  # unique keys detect in-sample target leakage
    d["Flight Distance"] = 100 + np.arange(n) % 4
    d["Arrival Delay in Minutes"] = np.where(np.arange(n) % 7 == 0, np.nan, np.arange(n) % 10)
    return pd.DataFrame(d)


def data():
    a, b, t = rows(), rows(16), rows(13)
    y = pd.Series(np.arange(len(a)) % 2)
    original = rows(22).assign(satisfaction=np.arange(22) % 2)
    return a, b, t, y, original


def test_full_fits_only_a_and_roundtrips_state():
    a, b, t, y, original = data()
    xa, xb, xt, state = prepare_recipe(a, b, t, y, original, 42)
    changed_b, changed_t = b.copy(), t.copy()
    changed_b.loc[0, ["Class", "Flight Distance"]] = ["never-seen", 999999]
    changed_t.loc[0, ["Class", "Flight Distance"]] = ["unseen-test", 888888]
    xa2, _, _, _ = prepare_recipe(a, changed_b, changed_t, y, original, 42)
    assert_frame_equal(xa, xa2)
    assert_frame_equal(xb, transform_recipe(b, pickle.loads(pickle.dumps(state))))
    assert_frame_equal(xt, transform_recipe(t, state))
    assert list(xa) == list(xb) == list(xt)
    assert len([c for c in xa if c.startswith("org_")]) == 21
    assert len([c for c in xa if c.startswith("fdm_")]) == 20
    assert len([c for c in xa if c in {f"{n}_d-4" for n in NUMS}]) == 4
    assert len([c for c in xa if c.startswith("te2_")]) == len(PAIRS)
    assert xa["Arrival Delay in Minutes"].iloc[0] == 0
    unseen = transform_recipe(changed_b, state)
    assert unseen["fd_cnt"].iloc[0] == 0
    assert np.isfinite(unseen.select_dtypes(include="number").to_numpy()).all()
    for c in CATS:
        assert list(xa[c].cat.categories) == list(xb[c].cat.categories)
    assert pd.isna(unseen["Class"].iloc[0])


def test_fitted_rarity_is_frozen_without_transform_log(monkeypatch):
    a, b, t, y, original = data()
    xa, xb, xt, state = prepare_recipe(a, b, t, y, original, 42)
    assert "rarity_maps" in state
    assert "rarity_fallback" in state
    legacy = {k: v for k, v in state.items() if k not in ("rarity_maps", "rarity_fallback")}
    assert_frame_equal(xb, transform_recipe(b, legacy))
    assert_frame_equal(xt, transform_recipe(t, legacy))
    unusual = b.copy()
    unusual.loc[0, "Age"] = 999999
    unusual.loc[1, "Class"] = "unseen"
    unusual.loc[2, "Seat comfort"] = np.nan
    expected = transform_recipe(unusual, legacy)
    assert np.isfinite(expected.filter(regex="^rar_").to_numpy()).all()
    frozen = pickle.loads(pickle.dumps(state))
    def forbid_log(*args, **kwargs):
        raise AssertionError("transform must not recompute rarity logarithms")
    monkeypatch.setattr(np, "log", forbid_log)
    assert_frame_equal(xb, transform_recipe(b, frozen))
    assert_frame_equal(xt, transform_recipe(t, frozen))
    assert_frame_equal(expected, transform_recipe(unusual, frozen))
    assert_frame_equal(xa.filter(regex="^rar_"), transform_recipe(a, frozen).filter(regex="^rar_"))
    assert_frame_equal(expected.iloc[[2, 0]].reset_index(drop=True), transform_recipe(unusual.iloc[[2, 0]], frozen))


def test_target_encodings_are_cross_fitted_and_unseen_falls_back():
    a, b, t, y, original = data()
    xa, xb, _, state = prepare_recipe(a, b, t, y, original, 42, mode="te_only")
    assert not any(c.startswith(("org_", "fdm_", "freq_", "rar_")) or "_d-" in c for c in xa)
    assert list(xa) == list(BASE) + [f"te_{c}" for c in BASE] + [f"te2_{l}|{r}" for l, r in PAIRS]
    assert not np.allclose(xa["te_Age"], y)
    assert xa["te_Age"].between(0, 1).all()
    assert_frame_equal(xb, transform_recipe(b, state))
    unseen = b.copy()
    unseen.loc[0, "Age"] = 999999
    unseen.loc[0, "Class"] = "unknown"
    result = transform_recipe(unseen, state)
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()
    assert result["te_Age"].iloc[0] == state["te"].target_mean_


def test_fold_counts_routes_and_original_priors_have_distinct_sources():
    a, b, t, y, original = data()
    xa, _, _, state = prepare_recipe(a, b, t, y, original, 42)
    assert xa["org_Age"].iloc[0] == 0  # original-only unsmoothed singleton
    assert xa["org_Age"].iloc[-1] == original.satisfaction.mean()
    assert xa["fd_cnt"].iloc[0] == (a["Flight Distance"] == 100).sum()
    assert xa["fdm_Age"].iloc[0] == a.loc[a["Flight Distance"] == 100, "Age"].mean()
    assert xa["fdd_Age"].iloc[0] == a.Age.iloc[0] - xa["fdm_Age"].iloc[0]
    assert xa["freq_Age"].iloc[0] == 1 / len(a)
    assert np.isclose(xa["rar_Age"].iloc[0], -np.log(1 / len(a)))
    assert "freq_org_Age" not in xa
    assert state["numeric_feature_order"] == list(xa.select_dtypes(include="number"))
    assert state["original_provenance"]["label_source"] == "overlap-pruned original satisfaction"
    assert state["original_provenance"]["rows"] == len(original)
