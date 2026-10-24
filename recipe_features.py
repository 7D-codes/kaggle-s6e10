"""Fold-trained adaptation of the public S6E10 route/digit/frequency recipe.

The caller must remove all original/competition full-feature overlaps before passing
``original``. No competition validation or test row contributes fitted statistics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold

from features import encode_original_target

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort",
           "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
BASE = CATS + NUMS + RATINGS
PAIRS = [("Class", "Type of Travel"), ("Customer Type", "Type of Travel"),
         ("Class", "Customer Type"), ("Gender", "Class"),
         ("Inflight wifi service", "Online boarding"),
         ("Inflight wifi service", "Type of Travel"),
         ("Online boarding", "Type of Travel"),
         ("Seat comfort", "Inflight entertainment"),
         ("Class", "Inflight wifi service"), ("Class", "Online boarding")]


def _raw(raw, levels):
    frame = raw[BASE].reset_index(drop=True).copy()
    frame["Arrival Delay in Minutes"] = frame["Arrival Delay in Minutes"].fillna(0)
    for c in CATS:
        frame[c] = pd.Categorical(frame[c].where(frame[c].isin(levels[c])), categories=levels[c])
    return frame


def _te_inputs(frame):
    return frame[BASE].astype(str)


def _pair_inputs(frame):
    return pd.DataFrame({f"{l}|{r}": frame[l].astype(str) + "_" + frame[r].astype(str)
                         for l, r in PAIRS})


def _prepared(raw, state):
    x = _raw(raw, state["levels"])
    if state["mode"] == "te_only":
        return x
    # Original-target priors are independent of competition labels and its rows
    # have already been disjointed from the entire competition population.
    for c in BASE:
        x[f"org_{c}"] = x[c].map(state["original_rates"][c]).astype("float32").fillna(state["original_mean"])
    key = "Flight Distance"
    coded = x[BASE].copy()
    for c in CATS:
        coded[c] = coded[c].cat.codes
    x["fd_cnt"] = coded[key].map(state["route_counts"]).fillna(0).astype("float32")
    for c in BASE:
        if c == key:
            continue
        mean = coded[key].map(state["route_means"][c]).fillna(state["route_fallback"][c]).astype("float32")
        x[f"fdm_{c}"] = mean
        x[f"fdd_{c}"] = (coded[c] - mean).astype("float32")
    for c in NUMS:
        v = x[c].fillna(0)
        for k in range(-4, 4):
            x[f"{c}_d{k}"] = (v // (10.0 ** k) % 10).astype("int8")
    frequency_columns = {}
    for c, counts in state["frequencies"].items():
        freq = x[c].map(counts).astype("float32").fillna(0)
        frequency_columns[f"freq_{c}"] = freq
        if "rarity_maps" in state:
            frequency_columns[f"rar_{c}"] = freq.map(state["rarity_maps"][c]).fillna(
                state["rarity_fallback"]).astype("float32")
        else:  # Previously sealed states did not store rarity outputs.
            frequency_columns[f"rar_{c}"] = (-np.log(freq.clip(lower=1 / state["training_rows"]))).astype("float32")
    return pd.concat([x, pd.DataFrame(frequency_columns, index=x.index)], axis=1)


def transform_recipe(raw, state):
    """Transform new raw rows with frozen fold statistics and fitted TE encoders."""
    x = _prepared(raw, state)
    for prefix, inputs, encoder in (("te_", _te_inputs(x), state["te"]),
                                    ("te2_", _pair_inputs(x), state["te2"])):
        values = encoder.transform(inputs)
        x = pd.concat([x, pd.DataFrame(values.astype("float32"),
                                      columns=[prefix + c for c in inputs], index=x.index)], axis=1)
    return x[state["feature_order"]]


def prepare_recipe(a, b, t, y, original, seed, mode="full"):
    """Return (train, validation, test, state), using only ``a/y`` for fold fits.

    ``original`` must be overlap-pruned by the caller. Validation/test labels
    are neither accepted nor inspected. Training target encodings are OOF;
    ``transform_recipe(a, state)`` is deliberately not identical to training TE.
    """
    if mode not in ("full", "te_only"):
        raise ValueError("mode must be 'full' or 'te_only'")
    if len(a) != len(y) or list(a.columns) != BASE and set(a.columns) != set(BASE):
        raise ValueError("Expected 21 raw features and one target per training row")
    levels = {c: a[c].dropna().unique().tolist() for c in CATS}
    state = {"mode": mode, "levels": levels, "training_rows": len(a),
             "numeric_feature_order": []}
    train = _raw(a, levels)
    if mode == "full":
        source = original[BASE + ["satisfaction"]].copy()
        state["original_provenance"] = {"label_source": "overlap-pruned original satisfaction",
                                        "rows": len(source)}
        source["Arrival Delay in Minutes"] = source["Arrival Delay in Minutes"].fillna(0)
        source["satisfaction"] = encode_original_target(source["satisfaction"])
        state["original_mean"] = float(source["satisfaction"].mean())
        state["original_rates"] = {
            c: source.groupby(c, observed=True, dropna=False)["satisfaction"].mean()
            for c in BASE}
        coded = train[BASE].copy()
        for c in CATS:
            coded[c] = coded[c].cat.codes
        grouped = coded.groupby("Flight Distance", dropna=False, observed=True)
        state["route_counts"] = grouped.size()
        state["route_means"] = grouped[[c for c in BASE if c != "Flight Distance"]].mean()
        state["route_fallback"] = coded.mean(numeric_only=True).to_dict()
        # Fit frequency counts after all non-frequency blocks, never from heldouts.
        state["frequencies"] = {}
        base_state = {**state, "frequencies": {}}
        prepared = _prepared(a, base_state)
        for c in prepared:
            if not c.startswith("org_"):
                state["frequencies"][c] = prepared[c].value_counts(normalize=True, dropna=False)
        # Freeze the fitted runtime's float32 frequency -> float32 rarity conversion.
        floor = 1 / state["training_rows"]
        zero = pd.Series([0], dtype="float32")
        state["rarity_fallback"] = np.float32(-np.log(zero.clip(lower=floor)).iloc[0])
        state["rarity_maps"] = {}
        for c, counts in state["frequencies"].items():
            frequencies = pd.Series(counts.to_numpy(dtype="float32")).drop_duplicates()
            rarities = (-np.log(frequencies.clip(lower=floor))).astype("float32")
            state["rarity_maps"][c] = dict(zip(frequencies, rarities))
    xa = _prepared(a, state)
    state["te"] = TargetEncoder(target_type="binary", smooth="auto",
                                cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    state["te2"] = TargetEncoder(target_type="binary", smooth="auto",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    labels = np.asarray(y)
    for prefix, inputs, encoder in (("te_", _te_inputs(xa), state["te"]),
                                    ("te2_", _pair_inputs(xa), state["te2"])):
        encoded = encoder.fit_transform(inputs, labels)
        xa = pd.concat([xa, pd.DataFrame(encoded.astype("float32"),
                                        columns=[prefix + c for c in inputs], index=xa.index)], axis=1)
    state["feature_order"] = list(xa)
    state["numeric_feature_order"] = list(xa.select_dtypes(include="number"))
    return xa, transform_recipe(b, state), transform_recipe(t, state), state
