"""Source-only synthetic checks for the frozen continuation comparison."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import rankdata


def module():
    assert importlib.util.find_spec('stack_kumo_continuation'), 'frozen three-arm runner is missing'
    import stack_kumo_continuation
    return stack_kumo_continuation


def test_thirty_native_fits_same_excluded_fold_and_manual_replay():
    s = module(); m = s.m
    rng = np.random.default_rng(53); y = np.arange(400) % 2
    x = rng.normal(size=(400, 14)) + y[:, None] * .4
    folds = m.original_folds(y)
    prediction, cell, models, states = s.paired_fit(y, folds, [x[:, :k] for k in (12, 13, 14)])
    assert prediction.shape == (400, 3) and len(models) == len(states) == 30
    for fold, half, fitting, validation in m.cells(y, folds):
        assert not np.intersect1d(fitting, validation).size
        assert np.all(folds[fitting] == fold)
        for arm in range(3):
            state = next(r for r in states if (r['fold'], r['half'], r['arm']) == (fold, half, s.ARMS[arm]))
            assert state['fit_rows_sha256'] == m.vector_hash(fitting, '<i8')
            assert state['validation_rows_sha256'] == m.vector_hash(validation, '<i8')
            expected = expit(x[validation, :12 + arm] @ state['coef'] + state['intercept'])
            np.testing.assert_allclose(prediction[validation, arm], expected, rtol=1e-6, atol=1e-7)
            assert np.ptp(expected) > 1e-6
            assert state['native_state']['classes'] == [0, 1]
            assert state['native_state']['n_features_in'] == 12 + arm
            from sklearn.linear_model import LogisticRegression
            restored = LogisticRegression(**m.CONFIG)
            restored.coef_ = np.array([state['coef']]); restored.intercept_ = np.array([state['intercept']])
            restored.classes_ = np.array(state['native_state']['classes'])
            restored.n_features_in_ = state['native_state']['n_features_in']
            np.testing.assert_allclose(m.replay(restored, x[validation, :12 + arm]), expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(cell[validation], 2 * fold + half)


@pytest.mark.parametrize('fail_arm', [0, 1])
def test_both_mandatory_gates_required(fail_arm):
    s = module()
    good = dict(primary={'gain': .00002}, folds=[{'gain': .1}] * 4 + [{'gain': -.1}], bootstrap={'interval95': [.000001, .1]})
    bad = dict(good, primary={'gain': .0000199})
    checks = [good.copy(), good.copy()]; checks[fail_arm] = bad
    assert not s.promotion_gate(*checks, valid=True)
    assert s.promotion_gate(good, good, valid=True)
    assert not s.promotion_gate(good, good, valid=False)


def test_exact_fixed_correction_with_ties():
    s = module()
    baseline = np.array([.4, .2, .4, .9, .1]); control = np.linspace(.2, .8, 5); joint = control[::-1]
    expected = (rankdata((rankdata(baseline, method='average') - .5) / 5 + .05 * (joint - control), method='average') - .5) / 5
    np.testing.assert_array_equal(s.corrected_rank(baseline, control, joint), expected)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    s = module(); m = s.m
    n, nt = 200, 40; y = (np.arange(n) % 2).astype('uint8'); folds = m.original_folds(y)
    rng = np.random.default_rng(12)
    train = pd.DataFrame({'id': np.arange(n), 'satisfaction': y, 'Type of Travel': ['Business travel'] * n})
    test = pd.DataFrame({'id': np.arange(n, n + nt), 'Type of Travel': ['Business travel'] * nt})
    def csv(name, frame):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False); return m.sha256(path)
    def js(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); m.write_json(path, value); return m.sha256(path)
    base = dict(version=1, train='train.csv', test='test.csv', baseline='summit.csv', train_rows=n, test_rows=nt,
        fold_seed=42, fold_count=5, meta_split_seed=20261025, members=[], source_assurance='synthetic conditional assurance')
    for key, frame in [('train', train), ('test', test), ('baseline', pd.DataFrame({'id': test.id, 'satisfaction': np.linspace(.01, .99, nt)}))]:
        base[key + '_sha256'] = csv(base[key], frame)
    for key, value, dtype in [('train_id', train.id, '<i8'), ('test_id', test.id, '<i8'), ('target', y, 'uint8'), ('fold', folds, 'int8')]:
        base[key + '_sha256'] = m.vector_hash(value, dtype)
    for i, name in enumerate(m.MEMBER_NAMES):
        p, t = expit(rng.normal(size=n) + y * .3), expit(rng.normal(size=nt))
        member = dict(name=name, format='csv' if i == 9 else 'npy', representation='probability', fold_seed=42, fold_count=5, provenance='synthetic', source='synthetic')
        for key, values in [('oof', p), ('test', t)]:
            member[key] = name + '-' + key + '.' + member['format']; path = tmp_path / member[key]
            if i == 9:
                frame = pd.DataFrame(dict(id=train.id, target=y, fold=folds, prediction=values)) if key == 'oof' else pd.DataFrame(dict(id=test.id, satisfaction=values))
                frame.to_csv(path, index=False)
            else: np.save(path, values)
            member[key + '_sha256'] = m.sha256(path)
        base['members'].append(member)
    base11 = dict(version=1, members=list(m.MEMBER_NAMES) + ['original_only_xgb_teacher'], base_manifest='base.json', base_manifest_sha256=js('base.json', base), teacher_csv='teacher.csv',
        teacher_sha256=csv('teacher.csv', pd.DataFrame(dict(id=pd.concat([train.id, test.id], ignore_index=True), teacher_original_xgb=expit(rng.normal(size=n + nt))))),
        fold_sha256=base['fold_sha256'], meta_half_seed=20261025, logistic_parameters=m.CONFIG, source_assurance='synthetic')
    cat = dict(oof='cat-oof.csv', test='cat-test.csv', native_audit='cat-audit.json')
    cat['oof_sha256'] = csv(cat['oof'], pd.DataFrame(dict(id=train.id, satisfaction=y, fold=folds, catboost=expit(rng.normal(size=n)))))
    cat['test_sha256'] = csv(cat['test'], pd.DataFrame(dict(id=test.id, catboost=expit(rng.normal(size=nt)))))
    cat['native_audit_sha256'] = js(cat['native_audit'], dict(valid=True, complete_oof=True, complete_test=True, job='wasidyms/s6e10-codex-catboost-ctr-v1/1', output_sha256={'oof.csv': cat['oof_sha256'], 'test.csv': cat['test_sha256']}))
    control = dict(version=1, members=base11['members'] + ['own_catboost_ctr'], base11_manifest='base11.json', base11_manifest_sha256=js('base11.json', base11), catboost=cat,
        baseline_candidate='cat-baseline.csv', baseline_sha256=csv('cat-baseline.csv', pd.DataFrame(dict(id=test.id, satisfaction=np.linspace(.01, .99, nt)))),
        fold_sha256=base['fold_sha256'], meta_half_seed=20261025, logistic_parameters=m.CONFIG)
    control_sha = js(str(s.CONTROL), control)
    pair = s.PAIRED
    hashes = {'oof.csv': csv(str(pair / 'oof.csv'), pd.DataFrame(dict(id=train.id, satisfaction=y, fold=folds, raw=expit(rng.normal(size=n)), fe=expit(rng.normal(size=n))))),
              'test.csv': csv(str(pair / 'test.csv'), pd.DataFrame(dict(id=test.id, raw=expit(rng.normal(size=nt)), fe=expit(rng.normal(size=nt)))))}
    audit = dict(valid=True, complete_oof=True, complete_test=True, completed_folds=list(range(5)), job=s.JOB, kernel_id=141062725, output_sha256=hashes)
    js(str(s.AUDIT), audit)
    monkeypatch.setattr(s, 'CONTROL_SHA', control_sha); monkeypatch.setattr(s, 'BASE11_SHA', control['base11_manifest_sha256']); monkeypatch.setattr(s, 'BASE10_SHA', base11['base_manifest_sha256'])
    monkeypatch.setattr(m, 'TRAIN_ROWS', n); monkeypatch.setattr(m, 'TEST_ROWS', nt)
    (tmp_path / 'stack_public_context.py').write_bytes(Path(m.__file__).read_bytes())
    expected_cell = np.full(n, -1, dtype=np.int8)
    for fold, half, _, valid in m.cells(y, folds): expected_cell[valid] = 2 * fold + half
    csv(str(s.PRIOR / 'oof.csv'), pd.DataFrame(dict(id=train.id, target=y, fold=folds, cell=expected_cell, candidate=np.linspace(.1, .9, n))))
    csv(str(s.PRIOR / 'test.csv'), pd.DataFrame(dict(id=test.id, candidate=np.linspace(.1, .9, nt))))
    data = s.load_inputs(tmp_path)
    x = [m.features(data['oof'][:, :k], train['Type of Travel'], False) for k in (12, 13, 14)]
    pred, cell, models, _ = s.paired_fit(y, folds, x)
    xt = [m.features(data['test_predictions'][:, :k], test['Type of Travel'], False) for k in (12, 13, 14)]
    final = s.test_predictions(models, xt)
    csv(str(s.PRIOR / 'oof.csv'), pd.DataFrame(dict(id=train.id, target=y, fold=folds, cell=cell, candidate=pred[:, 0])))
    csv(str(s.PRIOR / 'test.csv'), pd.DataFrame(dict(id=test.id, candidate=final[:, 0])))
    return s, tmp_path, audit, js


@pytest.mark.parametrize('defect', ['valid', 'complete_oof', 'complete_test', 'completed_folds', 'job', 'kernel_id', 'hash', 'id', 'target', 'fold', 'partial', 'constant', 'nonfinite'])
def test_rejects_kumo_bad_receipt_or_rows_before_fit(inputs, monkeypatch, defect):
    s, root, audit, js = inputs
    if defect in audit:
        audit[defect] = False if defect in ('valid', 'complete_oof', 'complete_test') else [] if defect == 'completed_folds' else 'wrong'
    elif defect == 'hash': audit['output_sha256']['oof.csv'] = '0' * 64
    else:
        path = root / s.PAIRED / 'oof.csv'; f = pd.read_csv(path)
        if defect == 'partial': f = f.iloc[:-1]
        elif defect == 'constant': f['raw'] = .5
        elif defect == 'nonfinite': f.loc[0, 'fe'] = np.nan
        else: f.loc[0, {'id': 'id', 'target': 'satisfaction', 'fold': 'fold'}[defect]] = 999
        f.to_csv(path, index=False); audit['output_sha256']['oof.csv'] = s.m.sha256(path)
    js(str(s.AUDIT), audit)
    monkeypatch.setattr(s.m, 'fit_model', lambda *args: pytest.fail('bad artifact reached fitting'))
    with pytest.raises((ValueError, AssertionError)): s.run(root)
    metrics = json.loads((root / s.OUTPUT / 'metrics.json').read_text())
    assert metrics['status'] == 'failed' and not metrics['valid'] and not metrics['gate_passed']


@pytest.mark.parametrize('bad_contrast', ['fe_14_vs_13', 'joint_14_vs_12'])
def test_failed_gate_keeps_evidence_no_candidate(inputs, monkeypatch, bad_contrast):
    s, root, audit, js = inputs
    original = s.m.summarize; calls = []
    def summarize(*args):
        result = original(*args); calls.append(1)
        result['primary']['gain'] = -.1 if len(calls) == (1 if bad_contrast == 'fe_14_vs_13' else 2) else .1
        result['folds'] = [{'gain': .1}] * 5
        return result
    monkeypatch.setattr(s.m, 'summarize', summarize)
    monkeypatch.setattr(s.m, 'bootstrap_gain', lambda *args: {'interval95': [.01, .2]})
    result = s.run(root); out = root / s.OUTPUT
    assert result['valid'] and not result['gate_passed'] and result['fit_count'] == 30
    assert not (out / s.CANDIDATE).exists()
    assert len(json.loads((out / 'coefficients.json').read_text())) == 30
    with pytest.raises(FileExistsError): s.run(root)


@pytest.mark.parametrize('failure', [None, 'partial_write', 'post_publish', 'completion_receipt'])
def test_export_formula_and_failure_quarantine(inputs, monkeypatch, failure):
    s, root, audit, js = inputs; out = root / s.OUTPUT
    monkeypatch.setattr(s, 'promotion_gate', lambda *args, **kwargs: True)
    monkeypatch.setattr(s.m, 'bootstrap_gain', lambda *args: {'interval95': [.01, .2]})
    if failure == 'partial_write':
        original = pd.DataFrame.to_csv
        def write(frame, path, *args, **kwargs):
            if Path(path).name == s.INCOMPLETE:
                Path(path).write_text('partial'); raise OSError('synthetic write failure')
            return original(frame, path, *args, **kwargs)
        monkeypatch.setattr(pd.DataFrame, 'to_csv', write)
    elif failure == 'post_publish':
        original = s.m.sha256
        def digest(path):
            if Path(path).name == s.CANDIDATE: raise OSError('synthetic hash failure')
            return original(path)
        monkeypatch.setattr(s.m, 'sha256', digest)
    elif failure == 'completion_receipt':
        original = s.m.write_json
        def write(path, value):
            if isinstance(value, dict) and value.get('status') == 'complete': raise OSError('synthetic receipt failure')
            return original(path, value)
        monkeypatch.setattr(s.m, 'write_json', write)
    if failure:
        with pytest.raises(OSError, match='synthetic'): s.run(root)
        receipt = json.loads((out / 'metrics.json').read_text())
        assert not receipt['valid'] and not receipt['gate_passed'] and receipt['status'] == 'failed'
        assert not (out / s.CANDIDATE).exists() and (out / s.INCOMPLETE).exists()
        assert s.CANDIDATE not in receipt['artifacts_sha256']
        assert receipt['artifacts_sha256'][s.INCOMPLETE] == s.m.sha256(out / s.INCOMPLETE)
    else:
        result = s.run(root); assert result['gate_passed']
        test = pd.read_csv(out / 'test.csv'); baseline = pd.read_csv(root / 'cat-baseline.csv')
        expected = s.corrected_rank(baseline.satisfaction, test.control12, test.joint14)
        np.testing.assert_allclose(pd.read_csv(out / s.CANDIDATE).satisfaction, expected, rtol=0, atol=1e-15)
        assert s.m.sha256(out / 'catboost-baseline.csv') == s.m.sha256(root / 'cat-baseline.csv')


def test_missing_source_clone_artifacts_fail_clearly(tmp_path):
    s = module()
    with pytest.raises(FileNotFoundError, match='kumo-continuation-control-manifest'): s.run(tmp_path)
    receipt = json.loads((tmp_path / s.OUTPUT / 'metrics.json').read_text())
    assert receipt['status'] == 'failed' and receipt['fit_count'] == 0


def test_partial_cell_coverage_rejected(monkeypatch):
    s = module(); y = np.arange(200) % 2; folds = s.m.original_folds(y); original = s.m.cells
    monkeypatch.setattr(s.m, 'cells', lambda *args: list(original(*args))[:-1])
    with pytest.raises(ValueError, match='coverage'): s.paired_fit(y, folds, [np.tile(np.arange(200)[:, None] / 200, (1, k)) for k in (12, 13, 14)])


@pytest.mark.parametrize('key', ['oof', 'test'])
def test_retained_control_prediction_drift_fails_without_candidate(inputs, key):
    s, root, audit, js = inputs
    path = root / s.PRIOR / (key + '.csv'); frame = pd.read_csv(path)
    frame['candidate'] = 1 - frame.candidate; frame.to_csv(path, index=False)
    with pytest.raises(AssertionError): s.run(root)
    receipt = json.loads((root / s.OUTPUT / 'metrics.json').read_text())
    assert receipt['fit_count'] == 30 and not receipt['valid'] and not receipt['gate_passed']
    assert not (root / s.OUTPUT / s.CANDIDATE).exists()


@pytest.mark.parametrize('defect', ['cat_id', 'cat_fold', 'cat_hash', 'teacher_hash', 'helper_hash', 'prior_id', 'escape'])
def test_frozen_input_guards_precede_fitting(inputs, monkeypatch, defect):
    s, root, audit, js = inputs
    if defect in ('cat_id', 'cat_fold'):
        path = root / 'cat-oof.csv'; frame = pd.read_csv(path)
        frame.loc[0, 'id' if defect == 'cat_id' else 'fold'] = 999
        frame.to_csv(path, index=False)
        control = json.loads((root / s.CONTROL).read_text())
        control['catboost']['oof_sha256'] = s.m.sha256(path)
        cat_audit = json.loads((root / 'cat-audit.json').read_text())
        cat_audit['output_sha256']['oof.csv'] = s.m.sha256(path)
        control['catboost']['native_audit_sha256'] = js('cat-audit.json', cat_audit)
        monkeypatch.setattr(s, 'CONTROL_SHA', js(str(s.CONTROL), control))
    elif defect in ('cat_hash', 'teacher_hash', 'helper_hash'):
        name = {'cat_hash': 'cat-oof.csv', 'teacher_hash': 'teacher.csv', 'helper_hash': 'stack_public_context.py'}[defect]
        with (root / name).open('a') as stream: stream.write('\nchanged')
    elif defect == 'prior_id':
        path = root / s.PRIOR / 'oof.csv'; frame = pd.read_csv(path); frame.loc[0, 'id'] = 999; frame.to_csv(path, index=False)
    else:
        control = json.loads((root / s.CONTROL).read_text()); control['baseline_candidate'] = '../outside.csv'
        monkeypatch.setattr(s, 'CONTROL_SHA', js(str(s.CONTROL), control))
    monkeypatch.setattr(s.m, 'fit_model', lambda *args: pytest.fail('invalid input reached fitting'))
    with pytest.raises(ValueError): s.run(root)
    assert not json.loads((root / s.OUTPUT / 'metrics.json').read_text())['valid']


def test_timeout_records_failure_and_restores_timers(inputs, monkeypatch):
    import signal
    s, root, audit, js = inputs
    previous = signal.getsignal(signal.SIGPROF)
    def expire(*args):
        signal.getsignal(signal.SIGPROF)(signal.SIGPROF, None)
    monkeypatch.setattr(s.m, 'fit_model', expire)
    with pytest.raises(TimeoutError, match='10800'): s.run(root)
    receipt = json.loads((root / s.OUTPUT / 'metrics.json').read_text())
    assert receipt['status'] == 'failed' and receipt['fit_count'] == 0 and not receipt['valid']
    assert signal.getsignal(signal.SIGPROF) == previous
    assert signal.getitimer(signal.ITIMER_PROF) == (0., 0.)


def test_cli_requires_explicit_mode_and_self_test_is_synthetic(tmp_path):
    import subprocess
    import sys
    s = module(); script = Path(s.__file__)
    no_mode = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert no_mode.returncode == 2 and '--run' in no_mode.stderr
    result = subprocess.run([sys.executable, str(script), '--root', str(tmp_path), '--self-test'], capture_output=True, text=True)
    assert result.returncode == 0 and 'synthetic thirty-fit' in result.stdout
    assert not (tmp_path / s.OUTPUT).exists()
