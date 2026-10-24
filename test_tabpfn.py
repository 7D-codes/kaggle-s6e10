"""Synthetic tests for bounded raw-feature TabPFN wiring; no torch/model download."""
import json
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
    classes_ = np.array([0, 1])
    def __init__(self, **kwargs): self.parameters = kwargs
    def fit(self, x, y):
        self.fit_x, self.fit_y = x.copy(), y.copy()
        return self
    def predict_proba(self, x):
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
    assert metadata['repeat_prediction']['rtol'] == 1e-6 and metadata['repeat_prediction']['atol'] == 1e-7
    assert metadata['hashes']['checkpoint'] == pilot.sha256(checkpoint)
    assert json.loads((output / 'metrics.json').read_text())['status'] == 'complete'
    original = (output / 'metrics.json').read_bytes()
    with pytest.raises(FileExistsError): pilot.run(args, model_factory=factory)
    assert (output / 'metrics.json').read_bytes() == original


def test_repeat_prediction_difference_fails_without_partial_complete_claim(tmp_path):
    train, test = frames(); data = tmp_path / 'data'; data.mkdir()
    train.to_csv(data / 'train.csv', index=False); test.to_csv(data / 'test.csv', index=False)
    checkpoint = tmp_path / 'model'; checkpoint.write_bytes(b'fake')
    class Drifting(Estimator):
        calls = 0
        def predict_proba(self, x):
            self.calls += 1
            result = super().predict_proba(x)
            if self.calls >= 3: result[:, 1] += 1e-4; result[:, 0] -= 1e-4
            return result
    args = SimpleNamespace(data=data, output=tmp_path / 'out', checkpoint=checkpoint)
    with pytest.raises(AssertionError): pilot.run(args, model_factory=Drifting)
    failed = json.loads((args.output / 'metrics.json').read_text())
    assert failed['status'] == 'failed'
    assert failed['device']['requested'] == 'cuda:0'
    assert failed['device']['peak_allocated_bytes'] is None
    assert not (args.output / 'pilot-validation.csv').exists()
