"""Synthetic tests for bounded raw-feature TabPFN wiring; no torch/model download."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import train_tabpfn as pilot


def frames(n=62500):
    train = pd.DataFrame({c: np.arange(n) % 17 for c in pilot.FEATURES})
    for c in pilot.CATEGORICAL:
        train[c] = np.where(np.arange(n) % 2, 'a', 'b')
    train.insert(0, 'id', np.arange(n))
    train['satisfaction'] = np.arange(n) % 2
    test = train.iloc[:1000].drop(columns='satisfaction').copy()
    test['id'] += n
    test.loc[0, 'Gender'] = 'unseen'
    train.loc[0, 'Arrival Delay in Minutes'] = np.nan
    return train, test


def test_prepare_categories_are_context_local_and_numeric_nan_survives():
    train, test = frames(20)
    context = train.iloc[:10].drop(columns=['id', 'satisfaction'])
    valid = train.iloc[10:].drop(columns=['id', 'satisfaction']).copy()
    valid.loc[10, 'Gender'] = 'validation-only'
    a, b, t, categories = pilot.prepare(context, valid, test.drop(columns='id'))
    assert np.isnan(a[0, pilot.FEATURES.index('Arrival Delay in Minutes')])
    assert b[0, pilot.FEATURES.index('Gender')] == -1
    assert t[0, pilot.FEATURES.index('Gender')] == -1
    assert categories['Gender'] == ['b', 'a']
    assert a.shape == (10, 21)


@pytest.mark.parametrize('defect', ['duplicate', 'overlap', 'label', 'schema', 'infinity', 'numeric', 'category', 'small'])
def test_invalid_input_is_rejected(defect):
    train, test = frames()
    if defect == 'duplicate': train.loc[1, 'id'] = train.loc[0, 'id']
    if defect == 'overlap': test.loc[0, 'id'] = train.loc[0, 'id']
    if defect == 'label': train.loc[0, 'satisfaction'] = 2
    if defect == 'schema': test['extra'] = 1
    if defect == 'infinity':
        train['Age'] = train.Age.astype(float)
        train.loc[0, 'Age'] = np.inf
    if defect == 'numeric': train['Age'] = 'wrong'
    if defect == 'category': test['Gender'] = 1
    if defect == 'small': train = train.iloc[:100]
    with pytest.raises(ValueError): pilot.validate_inputs(train, test)


@pytest.mark.parametrize('probabilities', [np.full((3, 2), .5), [[0, 1], [0, np.nan], [1, 0]],
    [[0, 1.1], [1, 0], [.5, .5]], [[.3, .6], [.2, .8], [.1, .9]], [[.2], [.4], [.6]]])
def test_probabilities_must_be_bounded_finite_normalized_nonconstant(probabilities):
    with pytest.raises(ValueError): pilot.probabilities(probabilities, 3)


class Estimator:
    fit_mode = 'fit_with_cache'
    classes_ = np.array([0, 1])
    def __init__(self, **kwargs): self.parameters, self.calls = kwargs, []
    def fit(self, x, y):
        self.fit_x, self.fit_y = x.copy(), y.copy()
        return self
    def predict_proba(self, x):
        self.calls.append(x.copy())
        p = .1 + .8 * x[:, pilot.FEATURES.index('Age')] / 16
        return np.column_stack([1 - p, p])


def test_pilot_fits_only_first_50000_fold0_context_and_preserves_outputs(tmp_path):
    train, test = frames()
    data = tmp_path / 'data'; data.mkdir()
    train.to_csv(data / 'train.csv', index=False); test.to_csv(data / 'test.csv', index=False)
    checkpoint = tmp_path / 'model.safetensors'; checkpoint.write_bytes(b'synthetic checkpoint')
    output = tmp_path / 'pilot'
    captured = []
    def factory(**kwargs):
        model = Estimator(**kwargs); captured.append(model); return model
    args = SimpleNamespace(data=data, output=output, checkpoint=checkpoint)
    metadata = pilot.run(args, model_factory=factory)
    context, validation = next(StratifiedKFold(5, shuffle=True, random_state=42).split(train, train.satisfaction))
    model = captured[0]
    np.testing.assert_array_equal(model.fit_y, train.satisfaction.iloc[context[:50000]])
    np.testing.assert_array_equal(model.fit_x[:, pilot.FEATURES.index('Age')], train.Age.iloc[context[:50000]])
    assert model.parameters == dict(pilot.MODEL_PARAMETERS, model_path=str(checkpoint), categorical_features_indices=[pilot.FEATURES.index(c) for c in pilot.CATEGORICAL])
    valid = pd.read_csv(output / 'pilot-validation.csv'); predictions = pd.read_csv(output / 'pilot-test.csv')
    assert list(valid) == ['id', 'fold', 'target', 'prediction']
    np.testing.assert_array_equal(valid.id, train.id.iloc[validation[:1000]])
    np.testing.assert_array_equal(valid.target, train.satisfaction.iloc[validation[:1000]])
    assert valid.fold.eq(0).all() and list(predictions) == ['id', 'satisfaction']
    assert metadata['status'] == 'complete' and metadata['full_oof'] is False
    assert metadata['context_rows'] == 50000 and metadata['validation_rows'] == metadata['test_rows'] == 1000
    assert metadata['submissions'] == 0
    dispatch = metadata['inference_dispatch']
    assert dispatch['block_rows'] == 128 and dispatch['tail_policy'] == 'repeat_last_real_row'
    assert dispatch['preserves_query_order'] is True and dispatch['native_boundary_asserted'] is True
    assert dispatch['native_calls'] == len(model.calls)
    assert dispatch['native_query_shapes'] == [[128, len(pilot.FEATURES)]]
    assert all(len(x) == 128 for x in model.calls)
    assert metadata['validation_prediction_runtime_seconds'] >= 0
    comparisons = metadata['native_calibration']['comparisons']
    assert {row['name'] for row in comparisons} == {
        'repeat64', 'cross1000_64', 'native128_anchor', 'permutation1000',
        *[f'subset{n}' for n in (1, 17, 63, 65, 127, 128, 129)],
        'repeated_row0_slots', 'repeated_row0_mixed',
        'repeated_row1_slots', 'repeated_row1_mixed'}
    assert all(row['passed'] and row['rtol'] == 1e-6 and row['atol'] == 1e-7
               and row['max_abs_error'] >= 0 for row in comparisons)
    assert metadata['fitted_preprocessing']['status'] == 'unavailable_synthetic_estimator'
    assert metadata['model_parameters']['inference_precision'] == 'float32'
    assert json.loads((output / 'metrics.json').read_text())['model_parameters']['inference_precision'] == 'float32'
    assert metadata['repeat_prediction']['rtol'] == 1e-6 and metadata['repeat_prediction']['atol'] == 1e-7
    assert metadata['hashes']['checkpoint'] == pilot.sha256(checkpoint)
    assert json.loads((output / 'metrics.json').read_text())['status'] == 'complete'
    same_checkpoint = tmp_path / 'same-model-other-path'; same_checkpoint.write_bytes(checkpoint.read_bytes())
    rerun = pilot.run(SimpleNamespace(data=data, output=tmp_path / 'rerun', checkpoint=same_checkpoint), model_factory=factory)
    assert len(metadata['hashes']['configuration']) == 64
    assert metadata['hashes']['configuration'] == rerun['hashes']['configuration']
    original = (output / 'metrics.json').read_bytes()
    with pytest.raises(FileExistsError): pilot.run(args, model_factory=factory)
    assert (output / 'metrics.json').read_bytes() == original


def test_batch_size_prediction_difference_fails_without_partial_complete_claim(tmp_path):
    train, test = frames(); data = tmp_path / 'data'; data.mkdir()
    train.to_csv(data / 'train.csv', index=False); test.to_csv(data / 'test.csv', index=False)
    checkpoint = tmp_path / 'model'; checkpoint.write_bytes(b'fake')
    class Drifting(Estimator):
        drift_calls = 0
        def predict_proba(self, x):
            self.drift_calls += 1
            result = super().predict_proba(x)
            if self.drift_calls >= 3: result[:, 1] += 1e-4; result[:, 0] -= 1e-4
            return result
    args = SimpleNamespace(data=data, output=tmp_path / 'out', checkpoint=checkpoint)
    with pytest.raises(AssertionError): pilot.run(args, model_factory=Drifting)
    failed = json.loads((args.output / 'metrics.json').read_text())
    assert failed['status'] == 'failed'
    comparison = failed['native_calibration']['comparisons'][-1]
    assert comparison['passed'] is False and comparison['max_abs_error'] > 1e-7
    assert failed['device']['requested'] == 'cuda:0'
    assert failed['device']['peak_allocated_bytes'] is None
    assert not (args.output / 'pilot-validation.csv').exists()


def test_backend_receives_fixed_float32_without_changing_serialized_config(monkeypatch):
    dtype = object()
    torch = SimpleNamespace(float32=dtype, cuda=SimpleNamespace(is_available=lambda: True,
        device_count=lambda: 1, set_device=lambda index: None, reset_peak_memory_stats=lambda index: None))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'tabpfn', SimpleNamespace(TabPFNClassifier=Estimator))
    monkeypatch.setattr(pilot, 'version', lambda name: '9.0.0')
    parameters = dict(pilot.MODEL_PARAMETERS)
    model = pilot.make_model(**parameters)
    assert model.parameters['inference_precision'] is dtype
    assert parameters['inference_precision'] == 'float32'
    assert json.loads(json.dumps(pilot.MODEL_PARAMETERS))['inference_precision'] == 'float32'
    assert {key: value for key, value in model.parameters.items() if key != 'inference_precision'} == {
        key: value for key, value in parameters.items() if key != 'inference_precision'}


class ShapeDrifting(Estimator):
    def predict_proba(self, x):
        result = super().predict_proba(x)
        drift = 1e-4 if len(x) != 128 else 0
        result[:, 1] += drift
        result[:, 0] -= drift
        return result


def test_canonical_dispatch_removes_shape_drift_with_order_and_tail_preserved():
    train, _ = frames(1000)
    x = train[pilot.FEATURES].copy()
    for name in pilot.CATEGORICAL: x[name] = 0
    x = x.to_numpy(dtype=float)
    model = ShapeDrifting()
    assert not np.allclose(model.predict_proba(x)[:64], model.predict_proba(x[:128])[:64],
                           rtol=1e-6, atol=1e-7)
    model.calls.clear()
    base = pilot.predict_fixed(model, x)
    for size in (1, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000):
        indices = np.random.default_rng(42).permutation(len(x))[:size]
        replay = pilot.predict_fixed(model, x[indices])
        np.testing.assert_allclose(replay, base[indices], rtol=1e-6, atol=1e-7)
    permutation = np.random.default_rng(42).permutation(len(x))
    np.testing.assert_allclose(pilot.predict_fixed(model, x[permutation])[np.argsort(permutation)], base)
    assert all(block.shape == (128, len(pilot.FEATURES)) for block in model.calls)
    tail = model.calls[7]
    np.testing.assert_array_equal(tail[:104], x[896:])
    np.testing.assert_array_equal(tail[104:], np.repeat(x[-1:], 24, axis=0))
    assert pilot.predict_fixed(model, x[:1]).shape == (1,)
    calls_before = len(model.calls)
    with pytest.raises(ValueError, match='nonempty'):
        pilot.predict_fixed(model, x[:0])
    assert len(model.calls) == calls_before


@pytest.mark.parametrize('defect', ['shape', 'nan', 'bounds', 'normalization'])
def test_canonical_dispatch_rejects_invalid_native_probabilities(defect):
    class Invalid(Estimator):
        def predict_proba(self, x):
            result = np.full((len(x), 2), .5)
            if defect == 'shape': return result[:, :1]
            if defect == 'nan': result[0, 0] = np.nan
            if defect == 'bounds': result[0] = [-.1, 1.1]
            if defect == 'normalization': result[0] = [.3, .6]
            return result
    with pytest.raises(ValueError): pilot.predict_fixed(Invalid(), np.zeros((1, 21)))


@pytest.mark.parametrize('defect', [None, 'slot', 'composition'])
def test_native_calibration_exercises_slot_and_composition_independence(defect):
    class Biased(ShapeDrifting):
        def predict_proba(self, x):
            result = super().predict_proba(x)
            if defect == 'slot': drift = np.arange(len(x)) * 1e-4
            elif defect == 'composition': drift = (x[:, 0].mean() % 5) * 1e-4
            else: drift = 0
            result[:, 1] += drift
            result[:, 0] -= drift
            return result
    x = np.column_stack([np.arange(1000) % 17] * len(pilot.FEATURES)).astype(float)
    model = Biased()
    prediction = pilot.predict_fixed(model, x)
    metadata = dict(native_calibration=dict(comparisons=[]), inference_dispatch={})
    if defect:
        with pytest.raises(AssertionError): pilot.verify_native(model, x, prediction, metadata)
        assert metadata['native_calibration']['comparisons'][-1]['passed'] is False
    else:
        pilot.verify_native(model, x, prediction, metadata)
        assert all(item['passed'] for item in metadata['native_calibration']['comparisons'])
    assert all(len(block) == 128 for block in model.calls)


@pytest.mark.parametrize('drift', [False, True])
def test_fitted_cpu_preprocessing_records_config_and_checks_composition_before_assert(drift):
    class Member:
        config = {'fitted': 'synthetic'}
        def transform_X_test(self, x):
            return x + (np.nanmean(x[:, 0]) if drift else 0)
    model = SimpleNamespace(executor_=SimpleNamespace(ensemble_members=[Member()]))
    x = np.arange(128 * 21, dtype=float).reshape(128, 21)
    x[0, 1] = np.nan
    metadata = {}
    if drift:
        with pytest.raises(AssertionError): pilot.check_fitted_preprocessing(model, x, metadata)
        assert metadata['fitted_preprocessing']['comparisons'][-1]['passed'] is False
    else:
        pilot.check_fitted_preprocessing(model, x, metadata)
        receipt = metadata['fitted_preprocessing']
        assert receipt['status'] == 'passed' and receipt['covers_gpu_preprocessing'] is False
        assert receipt['member_configs'] == [repr(Member.config)]
        assert {row['name'] for row in receipt['comparisons']} == {
            'member0_subset17', 'member0_padded17', 'member0_repeated_row0', 'member0_repeated_row1'}
        assert all(row['passed'] and row['max_abs_error'] == 0 for row in receipt['comparisons'])
        assert receipt['comparisons'][0]['gating'] is False
        assert all(row['gating'] for row in receipt['comparisons'][1:])


@pytest.mark.parametrize('fit_mode', [None, 'fit_preprocessors'])
def test_canonical_dispatch_rejects_models_without_explicit_fitted_cache(fit_mode):
    model = SimpleNamespace(predict_proba=lambda x: np.full((len(x), 2), .5))
    if fit_mode is not None: model.fit_mode = fit_mode
    with pytest.raises(ValueError, match='fit_with_cache'):
        pilot.predict_fixed(model, np.zeros((1, 21)))


def test_variable_cpu_subset_is_diagnostic_while_canonical_composition_is_gating():
    class Member:
        config = 'synthetic shape-only numerical drift'
        def transform_X_test(self, x): return x + (1e-9 if len(x) != 128 else 0)
    model = SimpleNamespace(executor_=SimpleNamespace(ensemble_members=[Member()]))
    metadata = {}
    pilot.check_fitted_preprocessing(model, np.arange(128 * 21, dtype=float).reshape(128, 21), metadata)
    receipt = metadata['fitted_preprocessing']
    diagnostic = receipt['comparisons'][0]
    assert receipt['status'] == 'passed'
    assert diagnostic['gating'] is False and diagnostic['passed'] is False
    assert diagnostic['within_original_tolerance'] is True and diagnostic['max_abs_error'] > 0
    assert all(row['passed'] for row in receipt['comparisons'][1:])
