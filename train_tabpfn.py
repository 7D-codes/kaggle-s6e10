"""Fixed fold-0 feasibility pilot. No full OOF estimate or competition submission."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CATEGORICAL = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
FEATURES = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
MODEL_PARAMETERS = dict(n_estimators=1, random_state=42, device='cuda:0',
    fit_mode='fit_with_cache', kv_cache_precision='int8', inference_precision='float32',
    memory_saving_mode=True, keep_cache_on_device=False)
QUERY_BLOCK_ROWS = 128


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_inputs(train, test):
    for frame, expected in ((train, ['id', *FEATURES, 'satisfaction']), (test, ['id', *FEATURES])):
        if frame.columns.has_duplicates or set(frame.columns) != set(expected):
            raise ValueError('Input schema must contain exactly id, 21 raw features and train target')
        ids = frame['id']
        if (not pd.api.types.is_integer_dtype(ids.dtype) or ids.isna().any() or ids.duplicated().any()):
            raise ValueError('IDs must be nonmissing unique integers')
        for name in FEATURES:
            values = frame[name]
            if name in CATEGORICAL:
                if not values.dropna().map(lambda value: isinstance(value, str)).all():
                    raise ValueError(f'Expected string categories: {name}')
            elif (not pd.api.types.is_numeric_dtype(values.dtype)
                  or pd.api.types.is_bool_dtype(values.dtype)
                  or np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any()):
                raise ValueError(f'Expected numeric values without infinity: {name}')
    if train.id.isin(test.id).any():
        raise ValueError('Train/test IDs overlap')
    labels = train['satisfaction']
    if (labels.isna().any() or not pd.api.types.is_numeric_dtype(labels.dtype)
            or set(labels.unique()) != {0, 1} or labels.value_counts().min() < 5):
        raise ValueError('Target must be binary 0/1 with at least five rows per class')
    if len(train) < 62500 or len(test) < 1000:
        raise ValueError('Pilot requires 50,000 context and 1,000 validation/test rows')


def prepare(context, validation, test):
    """Learn categorical vocabularies only from the rows passed to fit."""
    frames = [frame[FEATURES].copy() for frame in (context, validation, test)]
    categories = {}
    for name in CATEGORICAL:
        categories[name] = context[name].dropna().unique().tolist()
        mapping = {value: index for index, value in enumerate(categories[name])}
        for frame in frames:
            frame[name] = frame[name].map(mapping).fillna(-1)
    return *(frame.to_numpy(dtype=np.float64) for frame in frames), categories


def probabilities(raw, rows, *, nonconstant=True):
    values = np.asarray(raw, dtype=float)
    if (values.shape != (rows, 2) or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
            or not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-7)
            or (nonconstant and np.ptp(values[:, 1]) <= 1e-6)):
        raise ValueError('Expected finite bounded normalized nonconstant binary probabilities')
    return values[:, 1]


def native128(model, queries, dispatch=None):
    """Call the unmodified backend only at the frozen numerical execution shape."""
    assert queries.ndim == 2 and len(queries) == QUERY_BLOCK_ROWS
    if dispatch is not None:
        dispatch['native_calls'] = dispatch.get('native_calls', 0) + 1
        shape = list(queries.shape)
        shapes = dispatch.setdefault('native_query_shapes', [])
        if shape not in shapes:
            shapes.append(shape)
    raw = np.asarray(model.predict_proba(queries), dtype=float)
    probabilities(raw, QUERY_BLOCK_ROWS, nonconstant=False)
    return raw


def predict_fixed(model, queries, dispatch=None, *, nonconstant=False):
    """Canonical inference for fitted-cache, independent query rows; retain input order."""
    if queries.ndim != 2 or len(queries) == 0:
        raise ValueError('Expected a nonempty two-dimensional query array')
    if getattr(model, 'fit_mode', None) != 'fit_with_cache':
        raise ValueError('Canonical inference requires fit_with_cache')
    blocks = []
    for start in range(0, len(queries), QUERY_BLOCK_ROWS):
        block = queries[start:start + QUERY_BLOCK_ROWS]
        real_rows = len(block)
        if real_rows < QUERY_BLOCK_ROWS:
            block = np.concatenate([block, np.repeat(block[-1:], QUERY_BLOCK_ROWS - real_rows, axis=0)])
        blocks.append(native128(model, block, dispatch)[:real_rows])
    return probabilities(np.concatenate(blocks), len(queries), nonconstant=nonconstant)


def compare_predictions(metadata, name, actual, expected):
    # Write evidence before asserting so the failure receipt preserves the numerical gate.
    metadata['native_calibration']['comparisons'].append(dict(name=name,
        rows=len(actual), rtol=1e-6, atol=1e-7,
        max_abs_error=float(np.max(np.abs(actual - expected))),
        passed=bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-7))))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def verify_native(model, valid, prediction, metadata):
    """Acceptance is for fixed128 dispatch, not arbitrary native query batch sizes."""
    dispatch = metadata['inference_dispatch']
    small = valid[:64]
    first = predict_fixed(model, small, dispatch, nonconstant=True)
    second = predict_fixed(model, small, dispatch, nonconstant=True)
    compare_predictions(metadata, 'repeat64', first, second)
    compare_predictions(metadata, 'cross1000_64', first, prediction[:len(small)])
    metadata['repeat_prediction'] = dict(rows=len(small), nonconstant=True, rtol=1e-6, atol=1e-7,
        max_abs_error=float(np.max(np.abs(first - second))),
        max_batch_abs_error=float(np.max(np.abs(first - prediction[:len(small)]))))
    mixed = probabilities(native128(model, valid[:QUERY_BLOCK_ROWS], dispatch), QUERY_BLOCK_ROWS)
    compare_predictions(metadata, 'native128_anchor', predict_fixed(model, valid[:QUERY_BLOCK_ROWS], dispatch), mixed)
    rng = np.random.default_rng(42)
    for size in (1, 17, 63, 65, 127, 128, 129):
        positions = rng.permutation(len(valid))[:size]
        compare_predictions(metadata, f'subset{size}', predict_fixed(model, valid[positions], dispatch), prediction[positions])
    positions = rng.permutation(len(valid))
    permuted = predict_fixed(model, valid[positions], dispatch, nonconstant=True)
    compare_predictions(metadata, 'permutation1000', permuted[np.argsort(positions)], prediction)
    for row in (0, 1):
        repeated = np.repeat(valid[row:row + 1], QUERY_BLOCK_ROWS, axis=0)
        slots = predict_fixed(model, repeated, dispatch)
        compare_predictions(metadata, f'repeated_row{row}_slots', slots, np.repeat(slots[0], QUERY_BLOCK_ROWS))
        compare_predictions(metadata, f'repeated_row{row}_mixed', slots, np.repeat(mixed[row], QUERY_BLOCK_ROWS))


def check_fitted_preprocessing(model, valid, metadata):
    """Compare the exposed fitted CPU transform only; native gates also exercise GPU stages."""
    members = getattr(getattr(model, 'executor_', None), 'ensemble_members', None)
    receipt = metadata['fitted_preprocessing'] = dict(status='checking' if members else 'unavailable_synthetic_estimator',
        covers_gpu_preprocessing=False, member_configs=[], comparisons=[])
    if not members:
        return
    full = valid[:QUERY_BLOCK_ROWS]
    positions = np.random.default_rng(42).permutation(QUERY_BLOCK_ROWS)[:17]
    subset = full[positions]
    padded = np.concatenate([subset, np.repeat(subset[-1:], QUERY_BLOCK_ROWS - len(subset), axis=0)])
    for index, member in enumerate(members):
        receipt['member_configs'].append(repr(member.config))
        transformed = np.asarray(member.transform_X_test(full))
        cases = [('subset17', subset, positions), ('padded17', padded, np.r_[positions, np.repeat(positions[-1], QUERY_BLOCK_ROWS - len(subset))])]
        cases += [(f'repeated_row{row}', np.repeat(full[row:row + 1], QUERY_BLOCK_ROWS, axis=0),
                   np.repeat(row, QUERY_BLOCK_ROWS)) for row in (0, 1)]
        for name, queries, common in cases:
            actual, expected = np.asarray(member.transform_X_test(queries)), transformed[common]
            equal = np.array_equal(actual, expected, equal_nan=True)
            maximum = None
            if actual.shape == expected.shape:
                with np.errstate(invalid='ignore'):
                    difference = float(np.nanmax(np.abs(actual - expected), initial=0))
                if np.isfinite(difference):
                    maximum = difference
            gating = len(queries) == QUERY_BLOCK_ROWS
            receipt['comparisons'].append(dict(name=f'member{index}_{name}', rows=len(queries),
                passed=bool(equal), max_abs_error=maximum, equal_nan=True, gating=gating,
                within_original_tolerance=bool(actual.shape == expected.shape
                    and np.allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True))))
            if gating:
                if not equal:
                    receipt['status'] = 'failed'
                np.testing.assert_array_equal(actual, expected)
    receipt['status'] = 'passed'


def make_model(**parameters):
    # Lazy imports keep source-only software verification independent of CUDA/weights.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    from tabpfn import TabPFNClassifier
    if version('tabpfn') != '9.0.0':
        raise RuntimeError('Pilot requires tabpfn==9.0.0')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Pilot requires CUDA:0; CPU fallback is not authorized')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    parameters['inference_precision'] = torch.float32
    return TabPFNClassifier(**parameters)


def run(args, model_factory=None):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch_runtime = None
    metadata = dict(status='running', purpose='fold-0 feasibility only; not full OOF', full_oof=False,
        submissions=0, split_seed=42, model_seed=42, folds=5, fold=0,
        model_parameters=dict(MODEL_PARAMETERS), features=FEATURES, versions={'python': platform.python_version()},
        device=dict(requested='cuda:0', actual=None, peak_allocated_bytes=None, peak_reserved_bytes=None),
        inference_dispatch=dict(block_rows=QUERY_BLOCK_ROWS, tail_policy='repeat_last_real_row',
            native_boundary_asserted=True, preserves_query_order=True, native_calls=0, native_query_shapes=[]),
        native_calibration=dict(rng_seed=42, subset_rows=[1, 17, 63, 65, 127, 128, 129],
            repeated_validation_rows=[0, 1], arbitrary_native_batch_equivalence=False, comparisons=[]))
    metrics = output / 'metrics.json'
    metrics.write_text(json.dumps(metadata, indent=2))
    try:
        data, checkpoint = Path(args.data), Path(args.checkpoint)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError('A nonempty explicit local checkpoint is required')
        metadata['hashes'] = {name: sha256(path) for name, path in {
            'runner_source': Path(__file__), 'checkpoint': checkpoint,
            'train': data / 'train.csv', 'test': data / 'test.csv'}.items()}
        train, test = pd.read_csv(data / 'train.csv'), pd.read_csv(data / 'test.csv')
        validate_inputs(train, test)
        context, validation = next(StratifiedKFold(5, shuffle=True, random_state=42).split(train, train.satisfaction))
        context, validation = context[:50000], validation[:1000]
        if len(context) != 50000 or len(validation) != 1000:
            raise ValueError('Pilot split does not provide the exact required row counts')
        x, valid, testing, categories = prepare(train.iloc[context], train.iloc[validation], test.iloc[:1000])
        parameters = dict(MODEL_PARAMETERS, model_path=str(checkpoint),
            categorical_features_indices=[FEATURES.index(name) for name in CATEGORICAL])
        metadata.update(context_rows=len(context), validation_rows=len(validation), test_rows=len(testing),
            categorical_vocabularies=categories, model_parameters=parameters)
        configuration = {name: metadata[name] for name in ('split_seed', 'model_seed', 'folds', 'fold',
            'context_rows', 'validation_rows', 'test_rows', 'categorical_vocabularies', 'features')}
        configuration['model_parameters'] = {key: value for key, value in parameters.items() if key != 'model_path'}
        configuration['inference_dispatch'] = {key: metadata['inference_dispatch'][key] for key in
            ('block_rows', 'tail_policy', 'preserves_query_order', 'native_boundary_asserted')}
        metadata['hashes']['configuration'] = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        for name in ('numpy', 'pandas', 'scikit-learn', 'tabpfn', 'torch'):
            try: metadata['versions'][name] = version(name)
            except PackageNotFoundError: metadata['versions'][name] = None
        model = (model_factory or make_model)(**parameters)
        if model_factory is None:
            import torch as torch_runtime
            metadata['device'].update(actual=torch_runtime.cuda.get_device_name(0),
                visible_count=torch_runtime.cuda.device_count())
        else:
            metadata['device']['actual'] = 'synthetic estimator; no GPU verification'
        model.fit(x, train.satisfaction.iloc[context].to_numpy(dtype=np.int64))
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError('Expected classifier classes in [0, 1] order')
        check_fitted_preprocessing(model, valid, metadata)
        if model_factory is None and metadata['fitted_preprocessing']['status'] != 'passed':
            raise RuntimeError('Native fitted preprocessing must be exposed for verification')
        if torch_runtime is not None:
            torch_runtime.cuda.synchronize(0)
        prediction_started = time.monotonic()
        prediction = predict_fixed(model, valid, metadata['inference_dispatch'], nonconstant=True)
        if torch_runtime is not None:
            torch_runtime.cuda.synchronize(0)
        metadata['validation_prediction_runtime_seconds'] = time.monotonic() - prediction_started
        test_prediction = predict_fixed(model, testing, metadata['inference_dispatch'], nonconstant=True)
        verify_native(model, valid, prediction, metadata)
        metadata['pilot_validation_auc'] = float(roc_auc_score(train.satisfaction.iloc[validation], prediction))
        if torch_runtime is not None:
            torch_runtime.cuda.synchronize(0)
        pd.DataFrame(dict(id=train.id.iloc[validation].to_numpy(), fold=0,
            target=train.satisfaction.iloc[validation].to_numpy(), prediction=prediction)).to_csv(output / 'pilot-validation.csv', index=False)
        pd.DataFrame(dict(id=test.id.iloc[:1000].to_numpy(), satisfaction=test_prediction)).to_csv(output / 'pilot-test.csv', index=False)
        metadata['status'] = 'complete'
        return metadata
    except Exception as error:
        metadata.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if torch_runtime is not None:
            metadata['device'].update(peak_allocated_bytes=torch_runtime.cuda.max_memory_allocated(0),
                peak_reserved_bytes=torch_runtime.cuda.max_memory_reserved(0))
        metadata['runtime_seconds'] = time.monotonic() - started
        metrics.write_text(json.dumps(metadata, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
