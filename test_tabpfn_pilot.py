"""Inspect and execute only standard-library supervision, never remote training."""
import ast
import json
from pathlib import Path
import runpy
import shutil
import time

import pytest


@pytest.fixture
def generated(tmp_path):
    root = Path(__file__).resolve().parent
    for name in ('package_tabpfn_pilot.py', 'train_tabpfn.py'):
        shutil.copy2(root / name, tmp_path / name)
    runpy.run_path(str(tmp_path / 'package_tabpfn_pilot.py'), run_name='__main__')
    script = tmp_path / 'kaggle-tabpfn-pilot/train.py'
    namespace = runpy.run_path(str(script), run_name='inspect_only')
    return script, namespace


def test_package_embeds_exact_runner_and_private_single_gpu_contract(generated):
    script, namespace = generated
    assert namespace['RUNNER_SOURCE'] == (script.parents[1] / 'train_tabpfn.py').read_text()
    metadata = json.loads((script.parent / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'wasidyms/s6e10-codex-tabpfn-pilot'
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True
    assert metadata['enable_internet'] is True and metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['competition_sources'] == ['playground-series-s6e10']
    assert metadata['kernel_sources'] == metadata['dataset_sources'] == metadata['model_sources'] == []
    worker = namespace['WORKER_SOURCE']
    assert 'tabpfn==9.0.0' in worker
    assert 'b0662010dd17679509f42326e2f9ba37a6f58818' in worker
    assert 'ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3' in worker
    assert "CUDA_VISIBLE_DEVICES" in worker
    assert namespace['TOTAL_BUDGET_SECONDS'] == 1800
    ast.parse(worker)


def test_supervisor_includes_setup_and_kills_descendants_on_deadline(generated, tmp_path):
    _, namespace = generated
    output = tmp_path / 'timeout'; output.mkdir()
    worker = '''import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('descendant.pid').write_text(str(child.pid))
time.sleep(60)
'''
    started = time.monotonic()
    result = namespace['supervise'](output, worker, started + 1)
    assert time.monotonic() - started < 3
    assert result['status'] == 'timeout' and result['submissions'] == 0
    pid = int((output / 'descendant.pid').read_text())
    # Linux may briefly retain a killed grandchild as a zombie until init reaps it.
    stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().split()[2] == 'Z'
    assert json.loads((output / 'execution.json').read_text())['status'] == 'timeout'


def test_supervisor_records_failure_without_retry(generated, tmp_path):
    _, namespace = generated
    output = tmp_path / 'failure'; output.mkdir()
    result = namespace['supervise'](output, "raise RuntimeError('synthetic failure')", time.monotonic() + 3)
    assert result['status'] == 'failed' and result['returncode'] != 0
    assert result['submissions'] == 0
    assert 'synthetic failure' in (output / 'worker.log').read_text()


def test_supervisor_does_not_start_work_after_budget_expiry(generated, tmp_path):
    _, namespace = generated
    output = tmp_path / 'expired'; output.mkdir()
    result = namespace['supervise'](output, "raise AssertionError('must not execute')", time.monotonic() - 1)
    assert result['status'] == 'timeout'
    assert not (output / 'worker.log').exists()


def test_worker_rejects_competition_file_drift_before_model_fit(generated, tmp_path):
    import hashlib
    _, namespace = generated
    tree = ast.parse(namespace['WORKER_SOURCE'])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'verify_competition_inputs']
    assert functions, 'Worker must verify known competition inputs before GPU pilot'
    scope = {'hashlib': hashlib}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<worker-input-check>', 'exec'), scope)
    for name in ('train.csv', 'test.csv'): (tmp_path / name).write_text('drifted')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        scope['verify_competition_inputs'](tmp_path)
    worker = namespace['WORKER_SOURCE']
    assert '2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36' in worker
    assert 'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6' in worker
    assert worker.index('verify_competition_inputs(inputs[0].parent)') < worker.index("str(runner), '--data'")


def test_supervisor_preserves_failure_receipt_when_stage_json_is_interrupted(generated, tmp_path):
    _, namespace = generated
    output = tmp_path / 'broken-stage'; output.mkdir()
    worker = "from pathlib import Path\nPath('stage.json').write_text('{')\nraise RuntimeError('interrupted stage')"
    result = namespace['supervise'](output, worker, time.monotonic() + 3)
    assert result['status'] == 'failed'
    assert 'worker_stage_error' in result
    assert json.loads((output / 'execution.json').read_text())['status'] == 'failed'
