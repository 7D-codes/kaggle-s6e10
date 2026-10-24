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
    fit_mode='fit_with_cache', kv_cache_precision='int8', inference_precision='autocast',
    memory_saving_mode=True, keep_cache_on_device=False)


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


def probabilities(raw, rows):
    values = np.asarray(raw, dtype=float)
    if (values.shape != (rows, 2) or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
            or not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-7)
            or np.ptp(values[:, 1]) <= 1e-6):
        raise ValueError('Expected finite bounded normalized nonconstant binary probabilities')
    return values[:, 1]


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
    return TabPFNClassifier(**parameters)


def run(args, model_factory=None):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch_runtime = None
    metadata = dict(status='running', purpose='fold-0 feasibility only; not full OOF', full_oof=False,
        submissions=0, split_seed=42, model_seed=42, folds=5, fold=0,
        model_parameters=dict(MODEL_PARAMETERS), features=FEATURES, versions={'python': platform.python_version()},
        device=dict(requested='cuda:0', actual=None, peak_allocated_bytes=None, peak_reserved_bytes=None))
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
        prediction = probabilities(model.predict_proba(valid), len(valid))
        test_prediction = probabilities(model.predict_proba(testing), len(testing))
        small = valid[:64]
        first = probabilities(model.predict_proba(small), len(small))
        second = probabilities(model.predict_proba(small), len(small))
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(first, prediction[:len(small)], rtol=1e-6, atol=1e-7)
        metadata['repeat_prediction'] = dict(rows=len(small), nonconstant=True, rtol=1e-6, atol=1e-7,
            max_abs_error=float(np.max(np.abs(first - second))),
            max_batch_abs_error=float(np.max(np.abs(first - prediction[:len(small)]))))
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
