"""Synthetic-only tests for the frozen public context experiment."""
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score


def module():
    assert importlib.util.find_spec('stack_public_context') is not None, 'frozen context runner is missing'
    import stack_public_context
    return stack_public_context


def test_same_original_fold_halves_cover_once():
    m = module()
    y = np.arange(200) % 2
    folds = m.original_folds(y)
    seen = np.zeros(len(y), int)
    for fold, half, train, valid in m.cells(y, folds):
        assert not np.intersect1d(train, valid).size
        assert np.all(folds[train] == fold) and np.all(folds[valid] == fold)
        assert set(np.r_[train, valid]) == set(np.flatnonzero(folds == fold))
        assert set(y[train]) == set(y[valid]) == {0, 1}
        seen[valid] += 1
    np.testing.assert_array_equal(seen, 1)


def test_frozen_context_features_and_clipping():
    m = module()
    p = np.array([[0., .2, 1.], [.1, .8, .9]])
    x = m.features(p, np.array(['Personal Travel', 'Business travel']), contextual=False)
    z = m.features(p, np.array(['Personal Travel', 'Business travel']), contextual=True)
    assert x.min() == -30 and x.max() == 30
    np.testing.assert_array_equal(z[:, :3], x)
    np.testing.assert_array_equal(z[:, 3], [1, 0])
    np.testing.assert_allclose(z[0, 4:], x[0] - x[0].mean())
    np.testing.assert_array_equal(z[1, 4:], 0)
    with pytest.raises(ValueError, match='Travel'):
        m.features(p, np.array(['unknown', 'Personal Travel']), True)


@pytest.mark.parametrize('bad', [np.ones(4), np.array([.1, .2, np.nan, .4]),
    np.array([.1, .2, np.inf, .4]), np.array([.1, .2, -.1, .4]),
    np.array([.1, .2, 1.1, .4]), np.ones((4, 1)), np.array(['a'] * 4), np.linspace(.5, .5 + 1e-12, 4)])
def test_probability_rejections(bad):
    with pytest.raises(ValueError):
        module().probability(bad, 4, 'fixture')


def test_primary_is_pair_weighted_not_pooled_calibration():
    m = module()
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    cell = np.repeat([0, 1], 4)
    control = np.array([.1, .2, .1, .2, .8, .8, .9, .9])
    candidate = np.array([.1, .9, .1, .9, .2, .7, .6, .8])
    result = m.summarize(y, control, candidate, cell)
    assert result['primary']['control'] == 1
    assert result['primary']['gain'] < 0
    assert result['pooled']['gain'] > 0
    weights = np.array([r['pairs'] for r in result['cells']])
    assert result['primary']['candidate'] == np.average([r['candidate'] for r in result['cells']], weights=weights)


def test_bootstrap_paired_deterministic_zero_and_unequal_weights():
    m = module()
    y = np.tile([0, 1, 0, 1], 10)
    cell = np.repeat(np.arange(10), 4)
    p = np.tile([.1, .7, .3, .8], 10)
    assert m.bootstrap_gain(y, p, p, cell)['interval95'] == [0., 0.]
    q = p.copy(); q[1::4] = .2
    assert m.bootstrap_gain(y, p, q, cell) == m.bootstrap_gain(y, p, q, cell)


def test_native_manual_nonconstant_and_warning_failure(monkeypatch):
    m = module()
    x = np.arange(60, dtype=float).reshape(20, 3) / 60
    y = np.tile([0, 0, 1, 1], 5)
    model, receipt = m.fit_model(x, y)
    p = m.replay(model, x)
    assert np.ptp(p) > 0
    np.testing.assert_allclose(p, expit(x @ np.array(receipt['coef']) + receipt['intercept']), rtol=1e-6, atol=1e-7)
    def fail(self, *args):
        warnings.warn('forced synthetic nonconvergence', ConvergenceWarning)
    monkeypatch.setattr(m.LogisticRegression, 'fit', fail)
    with pytest.raises(ConvergenceWarning):
        m.fit_model(x, y)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    m = module()
    n, nt = 200, 40
    y = (np.arange(n) % 2).astype('uint8')
    rng = np.random.default_rng(123)
    train = pd.DataFrame({'id': np.arange(n), 'satisfaction': y,
        'Type of Travel': np.where(np.arange(n) % 3, 'Business travel', 'Personal Travel')})
    test = train.iloc[:nt].drop(columns='satisfaction').copy(); test['id'] += n
    train.to_csv(tmp_path / 'train.csv', index=False); test.to_csv(tmp_path / 'test.csv', index=False)
    baseline = pd.DataFrame({'id': test.id, 'satisfaction': np.linspace(.01, .99, nt)})
    baseline.to_csv(tmp_path / 'baseline.csv', index=False)
    folds = m.original_folds(y)
    manifest = dict(version=1, train='train.csv', test='test.csv', baseline='baseline.csv',
        train_rows=n, test_rows=nt, fold_seed=42, fold_count=5, meta_split_seed=20261025,
        train_id_sha256=m.vector_hash(train.id, '<i8'), test_id_sha256=m.vector_hash(test.id, '<i8'),
        target_sha256=m.vector_hash(y, 'uint8'), fold_sha256=m.vector_hash(folds, 'int8'),
        members=[], source_assurance='conditional on TFM author attestation; historical exposure remains')
    for name in ('train', 'test', 'baseline'):
        manifest[name + '_sha256'] = m.sha256(tmp_path / manifest[name])
    for i, name in enumerate(m.MEMBER_NAMES):
        p = expit(.8 * y + rng.normal(size=n)); t = expit(rng.normal(size=nt))
        fmt = 'csv' if i == 9 else 'npy'
        oof, pred = f'{name}-oof.{fmt}', f'{name}-test.{fmt}'
        if fmt == 'csv':
            pd.DataFrame(dict(id=train.id, target=y, fold=folds, prediction=p)).to_csv(tmp_path / oof, index=False)
            pd.DataFrame(dict(id=test.id, satisfaction=t)).to_csv(tmp_path / pred, index=False)
        else:
            np.save(tmp_path / oof, p); np.save(tmp_path / pred, t)
        member = dict(name=name, format=fmt, oof=oof, test=pred, oof_sha256=m.sha256(tmp_path / oof),
            test_sha256=m.sha256(tmp_path / pred), representation='probability', fold_seed=42, fold_count=5,
            provenance='author attested conditional source', source='https://example.org/source')
        manifest['members'].append(member)
    path = tmp_path / 'manifest.json'
    def save():
        path.write_text(json.dumps(manifest))
    save()
    monkeypatch.setattr(m, 'TRAIN_ROWS', n); monkeypatch.setattr(m, 'TEST_ROWS', nt)
    return m, tmp_path, manifest, path, save


def test_manifest_valid_and_numpy_never_allows_pickle(inputs, monkeypatch):
    m, root, manifest, path, save = inputs
    original = np.load
    calls = []
    def checked(*args, **kwargs):
        calls.append(kwargs.get('allow_pickle')); return original(*args, **kwargs)
    monkeypatch.setattr(m.np, 'load', checked)
    data = m.load_inputs(root, path)
    assert data['oof'].shape == (200, 10) and data['test_predictions'].shape == (40, 10)
    assert calls == [False] * 18


@pytest.mark.parametrize('defect', ['hash', 'id', 'target', 'fold', 'shape', 'finite', 'source', 'member', 'seed', 'category', 'baseline'])
def test_bad_manifest_or_member_rejected_before_any_fit(inputs, monkeypatch, defect):
    m, root, manifest, path, save = inputs
    member = manifest['members'][-1]
    if defect == 'hash': manifest['train_sha256'] = '0' * 64
    elif defect in ('id', 'target', 'fold'):
        p = root / member['oof']; frame = pd.read_csv(p)
        frame.loc[0, {'id': 'id', 'target': 'target', 'fold': 'fold'}[defect]] = 999
        frame.to_csv(p, index=False); member['oof_sha256'] = m.sha256(p)
    elif defect in ('shape', 'finite'):
        member = manifest['members'][0]; p = root / member['oof']
        a = np.full((200, 2), .5) if defect == 'shape' else np.r_[np.nan, np.linspace(.1, .9, 199)]
        np.save(p, a); member['oof_sha256'] = m.sha256(p)
    elif defect == 'source': manifest['source_assurance'] = ''
    elif defect == 'member': manifest['members'].pop()
    elif defect == 'seed': manifest['fold_seed'] = 43
    elif defect == 'category':
        p = root / manifest['test']; f = pd.read_csv(p); f.loc[0, 'Type of Travel'] = 'unknown'
        f.to_csv(p, index=False); manifest['test_sha256'] = m.sha256(p)
    elif defect == 'baseline':
        p = root / manifest['baseline']; f = pd.read_csv(p).iloc[::-1]
        f.to_csv(p, index=False); manifest['baseline_sha256'] = m.sha256(p)
    save()
    monkeypatch.setattr(m, 'fit_model', lambda *a: pytest.fail('invalid input reached fit'))
    output = root / 'failed'
    with pytest.raises(ValueError): m.run(root, path, output)
    assert json.loads((output / 'metrics.json').read_text())['status'] == 'failed'


def test_complete_run_failed_gate_keeps_oof_coefficients_no_promotion(inputs, monkeypatch):
    m, root, manifest, path, save = inputs
    monkeypatch.setattr(m, 'screening_gate', lambda metrics: False)
    out = root / 'run'
    result = m.run(root, path, out)
    assert result['status'] == 'complete' and result['fit_count'] == 20
    assert len(result['cells']) == 10 and len(result['folds']) == 5
    assert result['source_assurance'] == manifest['source_assurance']
    assert len(json.loads((out / 'coefficients.json').read_text())) == 20
    oof = pd.read_csv(out / 'oof.csv')
    assert oof.cell.nunique() == 10 and len(oof) == 200 and not oof.isna().any().any()
    assert not (out / 'summit-candidate.csv').exists()
    with pytest.raises(FileExistsError): m.run(root, path, out)


def test_passed_gate_exports_rank_formula_and_archived_baseline(inputs, monkeypatch):
    m, root, manifest, path, save = inputs
    monkeypatch.setattr(m, 'screening_gate', lambda metrics: True)
    out = root / 'pass'
    m.run(root, path, out)
    from scipy.stats import rankdata
    frame = pd.read_csv(out / 'test.csv')
    baseline = pd.read_csv(root / manifest['baseline'])
    original_rank = (rankdata(baseline.satisfaction, method='average') - .5) / len(baseline)
    expected = (rankdata(original_rank + .05 * (frame.context - frame.control), method='average') - .5) / len(baseline)
    np.testing.assert_array_equal(np.loadtxt(out / 'summit-candidate.csv', delimiter=',', skiprows=1, usecols=1), expected)
    assert m.sha256(out / 'summit-baseline.csv') == manifest['baseline_sha256']


def test_gate_and_runtime_projection():
    m = module()
    metrics = {'primary': {'gain': .00005}, 'folds': [{'gain': .01}] * 4 + [{'gain': -.01}],
        'cells': [{'gain': .01}] * 8 + [{'gain': -.01}] * 2,
        'bootstrap': {'interval95': [.000001, .001]}, 'valid': True}
    assert m.screening_gate(metrics)
    metrics['valid'] = False; assert not m.screening_gate(metrics)
    with pytest.raises(TimeoutError): m.check_projection(2041, 2041)
    m.check_projection(2000, 2000)


def test_missing_manifest_failure_receipt(tmp_path):
    m = module()
    output = tmp_path / 'out'
    with pytest.raises(FileNotFoundError): m.run(tmp_path, tmp_path / 'missing.json', output)
    assert json.loads((output / 'metrics.json').read_text())['status'] == 'failed'


def test_fit_failure_receipt_and_no_complete_or_promoted_output(inputs, monkeypatch):
    m, root, manifest, path, save = inputs
    def fail(*args):
        raise ConvergenceWarning('synthetic failure')
    monkeypatch.setattr(m, 'fit_model', fail)
    out = root / 'convergence'
    with pytest.raises(ConvergenceWarning): m.run(root, path, out)
    receipt = json.loads((out / 'metrics.json').read_text())
    assert receipt['status'] == 'failed' and receipt['fit_count'] == 0
    assert not (out / 'summit-candidate.csv').exists()


def test_native_replay_rejects_constant_and_bad_normalization():
    m = module()
    class Constant:
        coef_ = np.zeros((1, 1)); intercept_ = np.zeros(1)
        def predict_proba(self, x): return np.full((len(x), 2), .5)
    with pytest.raises(ValueError, match='nonconstant'): m.replay(Constant(), np.arange(10)[:, None])
    class Unnormalized(Constant):
        def predict_proba(self, x): return np.column_stack([np.linspace(.1, .2, len(x)), np.linspace(.2, .3, len(x))])
    with pytest.raises(ValueError, match='normalized'): m.replay(Unnormalized(), np.arange(10)[:, None])


@pytest.mark.parametrize('key', ['train_id_sha256', 'test_id_sha256', 'target_sha256', 'fold_sha256'])
def test_canonical_vector_hashes_are_verified(inputs, key):
    m, root, manifest, path, save = inputs
    manifest[key] = '0' * 64; save()
    with pytest.raises(ValueError, match='canonical vector SHA256'): m.load_inputs(root, path)


def test_coefficient_receipts_identify_exact_same_fold_fit_and_validation(inputs, monkeypatch):
    m, root, manifest, path, save = inputs
    monkeypatch.setattr(m, 'screening_gate', lambda metrics: False)
    out = root / 'identity'; m.run(root, path, out)
    coefs = json.loads((out / 'coefficients.json').read_text())
    oof = pd.read_csv(out / 'oof.csv'); data = m.load_inputs(root, path)
    for fold, half, fit_rows, valid_rows in m.cells(data['y'], data['folds']):
        records = [r for r in coefs if r['fold'] == fold and r['half'] == half]
        assert len(records) == 2
        for record in records:
            assert record['fit_rows_sha256'] == m.vector_hash(fit_rows, '<i8')
            assert record['validation_rows_sha256'] == m.vector_hash(valid_rows, '<i8')
            contextual = record['arm'] == 'context'
            x = m.features(data['oof'][valid_rows], data['train']['Type of Travel'].iloc[valid_rows], contextual)
            manual = expit(x @ np.array(record['coef']) + record['intercept'])
            np.testing.assert_allclose(oof[record['arm']].iloc[valid_rows], manual, rtol=1e-6, atol=1e-7)


def test_bootstrap_matches_fixed_pair_weighted_stratified_paired_oracle():
    m = module()
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    cell = np.r_[np.zeros(4, int), np.ones(6, int)]
    control = np.array([.2, .5, .4, .9, .1, .3, .6, .2, .7, .8])
    context = np.array([.3, .4, .5, .9, .2, .4, .6, .3, .5, .8])
    rng = np.random.default_rng(314159); values = []
    for _ in range(200):
        gains = []
        for k in range(2):
            a = np.flatnonzero((cell == k) & (y == 0)); b = np.flatnonzero((cell == k) & (y == 1))
            rows = np.r_[rng.choice(a, len(a)), rng.choice(b, len(b))]
            gains.append(roc_auc_score(y[rows], context[rows]) - roc_auc_score(y[rows], control[rows]))
        values.append(np.average(gains, weights=[4, 9]))
    result = m.bootstrap_gain(y, control, context, cell)
    assert result['fixed_cell_pair_weights'] == [4, 9]
    np.testing.assert_array_equal(result['interval95'], np.percentile(values, [2.5, 97.5]))
