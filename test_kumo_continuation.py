"""Source-only continuation checks; synthetic fixtures never construct native Kumo."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from test_kumo_pair import synthetic_run


ROOT = Path(__file__).resolve().parent


def test_runner_defaults_and_original_numerical_definitions():
    m = importlib.import_module('train_kumo_pair')
    assert (getattr(m, 'FIRST_PAIR_SECONDS', None), getattr(m, 'TOTAL_BUDGET_SECONDS', None)) == (3600, 28800)
    definitions = [n for n in ast.parse((ROOT / 'train_kumo_pair.py').read_text()).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != 'run']
    digest = hashlib.sha256(ast.dump(ast.Module(body=definitions, type_ignores=[])).encode()).hexdigest()
    assert digest == '4cf2a5504b869e5abd11a98c2a30acf5789fcff85576c07228660ed745ceae6f'
    run = next(n for n in ast.parse((ROOT / 'train_kumo_pair.py').read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == 'run')
    names = {'select_context', 'fit_preprocessor', 'transform', 'StratifiedKFold', 'model_factory', 'NativeModel',
             'predict_fixed', 'verify_native', 'verify_raw_replay', 'compare_predictions', 'roc_auc_score',
             'validate_complete', 'validate_inputs'}
    calls = [ast.dump(n) for n in ast.walk(run) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Name) and n.func.id in names) or (isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'model'))]
    assert hashlib.sha256('\n'.join(calls).encode()).hexdigest() == '7dc3427de8f898f5c3e581fa548b1804d0c24f76142b5ad0ef917acd8f4ee15d'
    assert hashlib.sha256((ROOT / 'package_kumo_pair.py').read_bytes()).hexdigest() == '31a15db42a58cfb688d081c02b27cd63c90b25b90bb39d820493a79b0a65c267'


def test_continuation_package_changes_only_caps_and_identity(tmp_path):
    assert (ROOT / 'package_kumo_continuation.py').is_file(), 'Continuation builder is missing'
    m = importlib.import_module('package_kumo_continuation')
    old = importlib.import_module('package_kumo_pair')
    output = m.package(tmp_path / 'continuation')
    script = (output / 'train.py').read_text()
    compile(script, str(output / 'train.py'), 'exec')
    scope = {'__name__': 'test'}
    exec(script, scope)
    default = (ROOT / 'train_kumo_pair.py').read_text()
    expected = default.replace('FIRST_PAIR_SECONDS = 3600', 'FIRST_PAIR_SECONDS = 5400').replace(
        'TOTAL_BUDGET_SECONDS = 28800', 'TOTAL_BUDGET_SECONDS = 24900')
    assert scope['RUNNER_SOURCE'] == expected
    runner_scope = {'__name__': 'test', '__file__': str(ROOT / 'train_kumo_pair.py')}
    exec(scope['RUNNER_SOURCE'], runner_scope)
    assert (runner_scope['FIRST_PAIR_SECONDS'], runner_scope['TOTAL_BUDGET_SECONDS']) == (5400, 24900)
    assert (scope['FIRST_PAIR_SECONDS'], scope['TOTAL_BUDGET_SECONDS']) == (5400, 24900)
    assignments = ast.parse(scope['WORKER_SOURCE']).body[:5]
    worker = {n.targets[0].id: ast.literal_eval(n.value) for n in assignments}
    assert worker == dict(SOURCES={'train_kumo_pair.py': expected,
        'train_tabpfn.py': (ROOT / 'train_tabpfn.py').read_text()}, INPUT_HASHES=old.INPUT_HASHES,
        SOURCE_COMMIT=old.SOURCE_COMMIT, CHECKPOINT_COMMIT=old.CHECKPOINT_COMMIT, CHECKPOINT_SHA=old.CHECKPOINT_SHA)
    assert scope['WORKER_SOURCE'].endswith(old.WORKER)
    assert script.endswith(old.SUPERVISOR.replace('FIRST_PAIR_SECONDS = 3600', 'FIRST_PAIR_SECONDS = 5400').replace(
        'TOTAL_BUDGET_SECONDS = 28800', 'TOTAL_BUDGET_SECONDS = 24900'))
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata == dict(id='wasidyms/s6e10-codex-kumo-continuation-v1', title='S6E10 Codex Kumo Continuation v1',
        code_file='train.py', language='python', kernel_type='script', is_private=True, enable_gpu=True,
        enable_internet=True, machine_shape='NvidiaTeslaT4', competition_sources=['playground-series-s6e10'],
        dataset_sources=[], kernel_sources=[], model_sources=[])
    with pytest.raises(FileExistsError):
        m.package(output)


def test_guarded_cap_replacements_fail_closed():
    assert (ROOT / 'package_kumo_continuation.py').is_file(), 'Continuation builder is missing'
    m = importlib.import_module('package_kumo_continuation')
    for source in ('FIRST_PAIR_SECONDS = 3600', 'FIRST_PAIR_SECONDS = 3600\n' * 2,
                   'FIRST_PAIR_SECONDS = 5400\nTOTAL_BUDGET_SECONDS = 28800'):
        with pytest.raises(ValueError, match='budget constants'):
            m.continuation_caps(source)


@pytest.mark.parametrize(('elapsed', 'passes'), [(3984, True), (3984.001, False), (5400, False)])
def test_setup_inclusive_economic_boundary(monkeypatch, tmp_path, elapsed, passes):
    m, args, factory = synthetic_run(monkeypatch, tmp_path)
    assert hasattr(m, 'FIRST_PAIR_SECONDS'), 'Runner budget constants are missing'
    monkeypatch.setattr(m, 'FIRST_PAIR_SECONDS', 5400)
    monkeypatch.setattr(m, 'TOTAL_BUDGET_SECONDS', 24900)
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10000.0)
    monkeypatch.setenv('KUMO_STARTED_MONOTONIC', str(10000.0 - elapsed))
    if passes:
        m.run(args, factory)
    else:
        with pytest.raises(TimeoutError, match='cap'):
            m.run(args, factory)
    output = Path(args.output)
    receipt = json.loads((output / 'metrics.json').read_text())
    assert receipt['first_pair_seconds'] == pytest.approx(elapsed)
    assert receipt['projected_total_seconds'] == pytest.approx(6.25 * elapsed)
    assert (output / 'first-fold-complete.json').exists() == passes
    assert (output / 'oof.csv').exists() == passes
    assert all(a['status'] == 'complete' for a in receipt['folds'][0]['arms'].values())
    assert (output / 'oof.partial.csv').is_file()


def test_separate_monotonic_stage_receipts_and_regenerated_pairs(monkeypatch, tmp_path):
    m, args, factory = synthetic_run(monkeypatch, tmp_path)
    constructed = []
    class Model(factory):
        def __init__(self, checkpoint, state, arm, fold):
            super().__init__(checkpoint, state, arm, fold)
            constructed.append((fold, arm))
        def transformed(self, values):
            return [np.asarray(values[:, 2:3], dtype=float)] * 2
    metadata = m.run(args, Model)
    assert 'timings' in metadata, 'Separate setup timing is missing'
    assert metadata['timings']['setup_seconds'] >= 0
    assert constructed == [(fold, arm) for fold in range(5) for arm in ('raw', 'fe')]
    for fold in metadata['folds']:
        assert fold['context_preparation_seconds'] >= 0
        for receipt in fold['arms'].values():
            assert set(receipt['timings']) == {'transform_seconds', 'fit_seconds', 'native_calibration_seconds',
                'validation_seconds', 'test_seconds', 'total_seconds'}
            assert all(value >= 0 for value in receipt['timings'].values())
            assert receipt['timings']['total_seconds'] >= sum(value for key, value in receipt['timings'].items() if key != 'total_seconds')
            assert receipt['native_calibration']['rtol'] == 1e-6 and receipt['native_calibration']['atol'] == 1e-7
            assert len(receipt['native_calibration']['comparisons']) == 21


def test_failed_stage_keeps_partial_timings_and_no_marker(monkeypatch, tmp_path):
    m, args, factory = synthetic_run(monkeypatch, tmp_path)
    class Fail(factory):
        def fit(self, x, y):
            if self.arm == 'fe':
                raise RuntimeError('synthetic fit failure')
    with pytest.raises(RuntimeError, match='fit failure'):
        m.run(args, Fail)
    output = Path(args.output)
    metadata = json.loads((output / 'metrics.json').read_text())
    failed = metadata['folds'][0]['arms']['fe']
    assert 'timings' in failed, 'Failure timing receipt is missing'
    assert failed['stage'] == 'fit' and failed['status'] == 'failed'
    assert failed['timings']['transform_seconds'] >= 0 and failed['timings']['fit_seconds'] >= 0
    assert failed['timings']['total_seconds'] >= failed['timings']['fit_seconds']
    assert (output / 'fold0/raw_test.csv').exists()
    assert not (output / 'first-fold-complete.json').exists()


@pytest.mark.parametrize('marker', [False, True])
def test_continuation_supervisor_kills_blocking_process_group(tmp_path, marker):
    assert (ROOT / 'package_kumo_continuation.py').is_file(), 'Continuation builder is missing'
    m = importlib.import_module('package_kumo_continuation')
    scope = {'__name__': 'test', 'RUNNER_SOURCE': 'synthetic'}
    exec(m.SUPERVISOR, scope)
    if marker:
        (tmp_path / 'paired').mkdir()
        (tmp_path / 'paired/first-fold-complete.json').write_text('{}')
    worker = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', "import time;from pathlib import Path;time.sleep(1);Path('escaped').write_text('bad')"])
Path('partial').write_text('kept')
Path('child.pid').write_text(str(child.pid))
time.sleep(20)
"""
    result = scope['supervise'](tmp_path, worker, time.monotonic() + (.25 if marker else 3),
        first_deadline=time.monotonic() + (.1 if marker else .25))
    assert result['status'] == 'timeout' and result['deadline_kind'] == ('total' if marker else 'first_pair')
    assert result['total_budget_seconds'] == 24900 and result['returncode'] != 0
    assert result['runtime_seconds'] < 2
    assert (tmp_path / 'partial').read_text() == 'kept'
    pid = int((tmp_path / 'child.pid').read_text())
    status = Path(f'/proc/{pid}/status')
    if status.exists():
        assert 'State:\tZ' in status.read_text(), 'Child survived process-group cleanup'
    assert json.loads((tmp_path / 'execution.json').read_text()) == result
