"""Frozen paired raw/route Kumo experiment; outputs evidence, never a submission."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, distribution, version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from train_tabpfn import (CATEGORICAL, FEATURES, compare_predictions, predict_fixed,
                         probabilities, sha256, verify_native)

SOURCE_COMMIT = 'fdc8372b510e19148f122681461438576916062a'
CHECKPOINT_COMMIT = '3c3e10bbdb590ace29e7026847f92db3c603096d'
CHECKPOINT_SHA = '1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617'
INPUT_HASHES = {'train.csv': '2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36',
                'test.csv': 'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'}
GROUP_COLUMNS = ['Class', 'Type of Travel', 'Customer Type', 'Age', 'Online boarding', 'Inflight wifi service']
ENGINEERED = ['route_mean_' + name for name in GROUP_COLUMNS] + ['route_count', 'route_target']
MODEL_PARAMETERS = dict(task='classification', size='small', device='cuda:0',
                        num_estimators=2, estimator_batch_size=1, precision='float16')


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'{Path(path).name} SHA256 mismatch')
    return actual


def validate_id_order(train, test):
    if not np.array_equal(train.id, np.arange(len(train))) or not np.array_equal(test.id, np.arange(len(train), len(train) + len(test))):
        raise ValueError('Official row IDs/order mismatch')


def validate_inputs(train, test):
    from train_tabpfn import validate_inputs as validate_raw
    validate_raw(train, test)
    validate_id_order(train, test)
    if len(train) != 699635 or len(test) != 299844:
        raise ValueError('Official competition row counts mismatch')


def select_context(outer_train, labels, rows=100000):
    selected, _ = train_test_split(np.asarray(outer_train), train_size=rows,
                                  stratify=np.asarray(labels)[outer_train], random_state=42)
    return selected


def route_keys(frame):
    return frame['Flight Distance'].map(lambda value: '__missing__' if pd.isna(value) else str(float(value)))


def target_statistics(keys, labels):
    labels = np.asarray(labels, dtype=float)
    prior = float(labels.mean())
    groups = pd.DataFrame({'route': np.asarray(keys), 'y': labels}).groupby('route').y.agg(['sum', 'count'])
    encoded = (groups['sum'] + 20 * prior) / (groups['count'] + 20)
    return encoded.to_dict(), prior


def fit_preprocessor(context, labels, fold):
    context = context[FEATURES].reset_index(drop=True)
    labels = np.asarray(labels)
    if len(context) != len(labels) or len(context) < 5 or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Context requires aligned binary labels and at least five rows')
    categories = {name: sorted(context[name].dropna().unique().tolist()) for name in CATEGORICAL}
    keys = route_keys(context)
    state = dict(categories=categories, schema={name: 'categorical' if name in CATEGORICAL else 'numerical'
                 for name in [*FEATURES, *ENGINEERED]}, means={}, routes={}, te_seed=314159 + fold, smoothing=20)
    for name in GROUP_COLUMNS:
        values = context[name]
        if name in CATEGORICAL:
            values = values.map({value: i for i, value in enumerate(categories[name])}).fillna(-1)
        values = values.astype(float)
        mean = values.mean()
        state['means'][name] = float(mean) if pd.notna(mean) else 0.0
        state['routes'][name] = values.groupby(keys).mean().dropna().to_dict()
    state['counts'] = {key: int(value) for key, value in keys.value_counts().items()}
    state['target'], state['prior'] = target_statistics(keys, labels)
    context_te = np.empty(len(context))
    for fit, held in KFold(5, shuffle=True, random_state=314159 + fold).split(context):
        mapping, prior = target_statistics(keys.iloc[fit], labels[fit])
        context_te[held] = keys.iloc[held].map(mapping).fillna(prior).to_numpy()
    return state, context_te


def transform(frame, state, arm, context_te=None):
    if arm not in ('raw', 'fe'):
        raise ValueError('Unknown paired arm')
    result = frame[FEATURES].copy()
    for name in CATEGORICAL:
        result[name] = pd.Categorical(result[name].where(result[name].isin(state['categories'][name])), categories=state['categories'][name])
    for name in FEATURES:
        if name not in CATEGORICAL:
            result[name] = result[name].astype(float)
    if arm == 'fe':
        keys = route_keys(frame)
        for name in GROUP_COLUMNS:
            result['route_mean_' + name] = keys.map(state['routes'][name]).fillna(state['means'][name])
        result['route_count'] = keys.map(state['counts']).fillna(0).astype(float)
        result['route_target'] = (keys.map(state['target']).fillna(state['prior']).to_numpy()
                                  if context_te is None else np.asarray(context_te))
    return result


def calibration_metadata():
    return dict(inference_dispatch=dict(block_rows=128, tail_policy='repeat_last_real_row', native_calls=0,
                native_query_shapes=[], preserves_query_order=True),
                native_calibration=dict(comparisons=[], rtol=1e-6, atol=1e-7))


def validate_complete(oof, testing, ids, labels, folds, test_ids, completed):
    if completed != list(range(5)):
        raise ValueError('Incomplete paired folds')
    for frame, key, expected in ((oof, 'id', ids), (oof, 'satisfaction', labels),
                                 (oof, 'fold', folds), (testing, 'id', test_ids)):
        if not np.array_equal(frame[key].to_numpy(), expected):
            raise ValueError(f'Incomplete or mismatched {key} coverage/order')
    for frame in (oof, testing):
        if frame.id.duplicated().any():
            raise ValueError('Duplicate output IDs')
        for arm in ('raw', 'fe'):
            p = frame[arm].to_numpy(dtype=float)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or np.ptp(p) <= 1e-6:
                raise ValueError('Invalid or constant paired output')


def native_state_json(value):
    """Archive processor tensors as plain JSON, with no executable deserialization."""
    import torch
    if isinstance(value, torch.Tensor):
        if hasattr(value, 'to_arrow'):
            return dict(type=type(value).__name__, arrow=value.to_arrow().to_pylist())
        data = value.detach().cpu().numpy()
        return dict(dtype=str(value.dtype), shape=list(value.shape),
                    data=np.where(np.isfinite(data), data, None).tolist() if data.dtype.kind == 'f' else data.tolist())
    if isinstance(value, dict):
        return {str(key): native_state_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native_state_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported native processor state: {type(value).__name__}')


class NativeModel:
    fit_mode = 'fit_with_cache'

    def __init__(self, checkpoint, state, arm, fold):
        # No native import or constructor before all immutable artifact guards pass.
        verify_hash(checkpoint, CHECKPOINT_SHA)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        import sdm
        native_distribution = distribution('structured-data-models')
        direct = json.loads(native_distribution.read_text('direct_url.json') or '{}')
        if direct.get('vcs_info', {}).get('commit_id') != SOURCE_COMMIT:
            raise ValueError('SDM source commit mismatch')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One CUDA:0 device required; no CPU fallback')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        torch.manual_seed(42 + fold)
        self.torch, self.sdm = torch, sdm
        self.generator = torch.Generator(device='cuda:0').manual_seed(42 + fold)
        self.state, self.arm = state, arm
        self.columns = FEATURES + (ENGINEERED if arm == 'fe' else [])
        self.model = sdm.models.KumoTabular(task='classification', size='small', pretrained=False, device='cuda:0')
        self.model.models['classification'].load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=True))

    def table(self, values):
        frame = pd.DataFrame(values, columns=self.columns)
        for name in self.columns:
            if name in CATEGORICAL:
                frame[name] = pd.Categorical(frame[name], categories=self.state['categories'][name])
            else:
                frame[name] = frame[name].astype(float)
        return self.sdm.TableTensor.from_pandas(frame, stypes={name: self.state['schema'][name] for name in self.columns}, device='cuda:0')

    def fit(self, values, labels):
        context = self.table(values)
        target = pd.DataFrame({'satisfaction': pd.Categorical(np.asarray(labels).astype(str), categories=['0', '1'])})
        target = self.sdm.TableTensor.from_pandas(target, stypes={'satisfaction': 'categorical'}, device='cuda:0')
        with self.torch.amp.autocast('cuda', self.torch.float16):
            self.model.fit(context, target, num_estimators=2, estimator_batch_size=1, generator=self.generator)

    def predict_proba(self, values):
        with self.torch.amp.autocast('cuda', self.torch.float16):
            output = self.model.predict(self.table(values))
        columns = list(output.numerical_columns)
        if len(columns) != 2 or set(columns) != {'0', '1'}:
            raise ValueError('Expected exactly native classes 0,1')
        return output.numerical[:, [columns.index('0'), columns.index('1')]].detach().float().cpu().numpy()

    def transformed(self, values):
        execution = self.model._cache['recipe_execution']
        return [member.x.numerical.detach().cpu().numpy() for member in execution.transform(self.table(values), None)]

    def archive(self):
        recipe = self.model._cache['recipe_execution'].recipe
        return dict(configuration=repr(recipe), features=native_state_json(recipe.features.state_dict()),
                    target=native_state_json(recipe.target.state_dict()), output=native_state_json(recipe.output.state_dict()))

    def clear(self):
        self.model.clear()
        self.torch.cuda.empty_cache()


def verify_raw_replay(model, raw_rows, state, arm, expected, metadata):
    independent = transform(raw_rows.copy(), json.loads(json.dumps(state)), arm).to_numpy()
    compare_predictions(metadata, 'independent_raw_transform', predict_fixed(model, independent, metadata['inference_dispatch'], nonconstant=True), expected)
    if hasattr(model, 'transformed'):
        base = model.transformed(independent[:128])
        appended = model.transformed(np.concatenate([independent[:128], independent[-17:]]))
        subset = model.transformed(independent[:17])
        for i, (before, after, small) in enumerate(zip(base, appended, subset, strict=True)):
            for name, actual, expected_transform in (
                (f'native_transform_append_member{i}', before, after[:len(before)]),
                (f'native_transform_subset_member{i}', before[:17], small)):
                metadata['native_calibration']['comparisons'].append(dict(name=name, rows=len(actual),
                    rtol=1e-6, atol=1e-7, equal_nan=True,
                    max_abs_error=float(np.nanmax(np.abs(actual - expected_transform), initial=0)),
                    passed=bool(np.allclose(actual, expected_transform, rtol=1e-6, atol=1e-7, equal_nan=True))))
                np.testing.assert_allclose(actual, expected_transform, rtol=1e-6, atol=1e-7, equal_nan=True)


def run(args, model_factory=None):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    supervisor_start = float(os.environ.get('KUMO_STARTED_MONOTONIC', start))
    metadata = dict(status='running', full_oof=False, submissions=0, completed_folds=[], folds=[],
                    model_parameters=MODEL_PARAMETERS, source_commit=SOURCE_COMMIT, checkpoint_commit=CHECKPOINT_COMMIT,
                    versions={'python': platform.python_version()}, hashes={})
    metrics = output / 'metrics.json'
    def save():
        metadata['runtime_seconds'] = time.monotonic() - start
        metrics.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    save()
    try:
        data, checkpoint = Path(args.data), Path(args.checkpoint)
        metadata['hashes'] = {'train': verify_hash(data / 'train.csv', INPUT_HASHES['train.csv']),
            'test': verify_hash(data / 'test.csv', INPUT_HASHES['test.csv']), 'checkpoint': verify_hash(checkpoint, CHECKPOINT_SHA),
            'runner': sha256(__file__), 'dispatch_source': sha256(Path(__file__).with_name('train_tabpfn.py'))}
        for name in ('numpy', 'pandas', 'scikit-learn', 'torch', 'structured-data-models', 'pyarrow', 'huggingface_hub'):
            try: metadata['versions'][name] = version(name)
            except PackageNotFoundError: metadata['versions'][name] = None
        train, test = pd.read_csv(data / 'train.csv'), pd.read_csv(data / 'test.csv')
        validate_inputs(train, test)
        y = train.satisfaction.to_numpy(dtype=np.int64)
        oof = pd.DataFrame({'id': train.id, 'satisfaction': y, 'fold': np.full(len(train), -1), 'raw': np.nan, 'fe': np.nan})
        test_sum = {arm: np.zeros(len(test)) for arm in ('raw', 'fe')}
        expected_folds = np.full(len(train), -1)
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))
        for fold, (_, validation) in enumerate(splits): expected_folds[validation] = fold
        for fold, (outer_train, validation) in enumerate(splits):
            selected = select_context(outer_train, y)
            fold_dir = output / f'fold{fold}'
            fold_dir.mkdir()
            np.save(fold_dir / 'context_indices.npy', selected, allow_pickle=False)
            state, te = fit_preprocessor(train.iloc[selected], y[selected], fold)
            (fold_dir / 'preprocessor.json').write_text(json.dumps(state, sort_keys=True, allow_nan=False))
            fold_receipt = dict(fold=fold, status='running', context_rows=len(selected), validation_rows=len(validation), test_rows=len(test),
                context_indices_sha256=sha256(fold_dir / 'context_indices.npy'), preprocessor_sha256=sha256(fold_dir / 'preprocessor.json'), arms={})
            metadata['folds'].append(fold_receipt); save()
            for arm in ('raw', 'fe'):
                receipt = calibration_metadata(); receipt.update(status='running', seed=42 + fold)
                fold_receipt['arms'][arm] = receipt; save()
                model = None
                try:
                    context = transform(train.iloc[selected], state, arm, te).to_numpy()
                    valid = transform(train.iloc[validation], state, arm).to_numpy()
                    testing = transform(test, state, arm).to_numpy()
                    receipt['configuration_sha256'] = hashlib.sha256(json.dumps(dict(model=MODEL_PARAMETERS, state=state, arm=arm, seed=42+fold,
                        context=fold_receipt['context_indices_sha256']), sort_keys=True).encode()).hexdigest()
                    model = (model_factory or NativeModel)(checkpoint, state, arm, fold)
                    model.fit(context, y[selected]); receipt['stage'] = 'native_calibration'; save()
                    anchor = predict_fixed(model, valid[:1000], receipt['inference_dispatch'], nonconstant=True)
                    verify_native(model, valid[:1000], anchor, receipt)
                    verify_raw_replay(model, train.iloc[validation[:1000]], state, arm, anchor, receipt)
                    if hasattr(model, 'archive'):
                        path = fold_dir / f'{arm}_native_preprocessor.json'
                        path.write_text(json.dumps(model.archive(), allow_nan=False))
                        receipt['native_preprocessor_sha256'] = sha256(path)
                    receipt['stage'] = 'validation'; save()
                    prediction = predict_fixed(model, valid, receipt['inference_dispatch'], nonconstant=True)
                    compare_predictions(receipt, 'full_validation_calibration_prefix', prediction[:len(anchor)], anchor)
                    pd.DataFrame({'id': train.id.iloc[validation], 'prediction': prediction}).to_csv(fold_dir / f'{arm}_validation.csv', index=False)
                    receipt['stage'] = 'test'; save()
                    test_prediction = predict_fixed(model, testing, receipt['inference_dispatch'], nonconstant=True)
                    pd.DataFrame({'id': test.id, 'prediction': test_prediction}).to_csv(fold_dir / f'{arm}_test.csv', index=False)
                    if model_factory is None:
                        import torch
                        receipt['device'] = dict(name=torch.cuda.get_device_name(0), peak_allocated=torch.cuda.max_memory_allocated(0), peak_reserved=torch.cuda.max_memory_reserved(0))
                    receipt.update(status='complete', stage='complete', hashes={name: sha256(fold_dir / f'{arm}_{name}.csv') for name in ('validation', 'test')})
                    oof.loc[validation, arm] = prediction
                    test_sum[arm] += test_prediction
                except Exception as error:
                    receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
                    raise
                finally:
                    if model is not None: model.clear()
                    save()
            oof.loc[validation, 'fold'] = fold
            fold_receipt['status'] = 'complete'
            metadata['completed_folds'].append(fold)
            oof.to_csv(output / 'oof.partial.csv', index=False)
            if fold == 0:
                elapsed = time.monotonic() - supervisor_start
                metadata['first_pair_seconds'] = elapsed
                metadata['projected_total_seconds'] = 1.25 * 5 * elapsed
                if elapsed > 3600 or metadata['projected_total_seconds'] > 28800:
                    raise TimeoutError('First pair time/projection exceeds frozen cap')
                # Supervisor requires both arms/full predictions/replay/archive before accepting this marker.
                (output / 'first-fold-complete.json').write_text(json.dumps(fold_receipt, allow_nan=False))
            save()
        testing = pd.DataFrame({'id': test.id, **{arm: test_sum[arm] / 5 for arm in ('raw', 'fe')}})
        validate_complete(oof, testing, train.id.to_numpy(), y, expected_folds, test.id.to_numpy(), metadata['completed_folds'])
        oof.to_csv(output / 'oof.csv', index=False); testing.to_csv(output / 'test.csv', index=False)
        metadata.update(status='complete', full_oof=True, auc={arm: float(roc_auc_score(y, oof[arm])) for arm in ('raw', 'fe')})
        metadata['hashes'].update(oof=sha256(output / 'oof.csv'), predictions_test=sha256(output / 'test.csv'))
    except Exception as error:
        metadata.update(status='failed', full_oof=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True); parser.add_argument('--checkpoint', required=True); parser.add_argument('--output', required=True)
    run(parser.parse_args())
