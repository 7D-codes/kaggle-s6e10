"""Create the private 1,710-second TabPFN FP32 feasibility package. Never launches or submits."""
import json
from pathlib import Path
import py_compile

WORKER = '''import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import hashlib, json, subprocess, sys, urllib.request
from pathlib import Path
root = Path.cwd()
deadline = float(os.environ['PILOT_DEADLINE_MONOTONIC'])
def remaining():
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Pilot total budget exhausted')
    return seconds

def stage(name):
    (root / 'stage.json').write_text(json.dumps({'stage': name, 'remaining_seconds': remaining()}))

def verify_competition_inputs(data):
    expected = {'train.csv': '2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36',
                'test.csv': 'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'}
    for name, digest in expected.items():
        with (data / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError(f'Competition {name} SHA256 mismatch')

stage('input_validation')
inputs = list(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv'))
if len(inputs) != 1:
    raise ValueError('Expected exactly one competition train.csv')
verify_competition_inputs(inputs[0].parent)
stage('setup')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
    '--upgrade-strategy', 'only-if-needed', 'tabpfn==9.0.0'], check=True, timeout=remaining())
stage('checkpoint_download')
url = 'https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/b0662010dd17679509f42326e2f9ba37a6f58818/tabpfn-v3.5-20260909.safetensors'
checkpoint = root / 'tabpfn-v3.5-20260909.safetensors'
with urllib.request.urlopen(url, timeout=min(60, remaining())) as response, checkpoint.open('xb') as stream:
    while True:
        remaining()
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        stream.write(chunk)
with checkpoint.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != 'ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3':
    raise ValueError('Official checkpoint SHA256 mismatch')
stage('pilot')
runner = root / 'train_tabpfn.py'
runner.write_text(RUNNER_SOURCE)
subprocess.run([sys.executable, str(runner), '--data', str(inputs[0].parent),
    '--checkpoint', str(checkpoint), '--output', str(root / 'pilot')], check=True, timeout=remaining())
metrics = json.loads((root / 'pilot/metrics.json').read_text())
if metrics['status'] != 'complete' or metrics['full_oof'] or metrics['submissions'] != 0:
    raise ValueError('Incomplete or invalid pilot receipt')
stage('complete')
'''

SUPERVISOR = '''import time
STARTED = time.monotonic()
import hashlib, json, os, signal, subprocess, sys
from pathlib import Path
TOTAL_BUDGET_SECONDS = 1710

def supervise(output, worker_source, deadline):
    started = time.monotonic()
    result = dict(status='timeout', submissions=0, returncode=None, total_budget_seconds=TOTAL_BUDGET_SECONDS)
    remaining = deadline - time.monotonic()
    if remaining > 0:
        worker = output / 'worker.py'
        worker.write_text(worker_source)
        environment = dict(os.environ, PILOT_DEADLINE_MONOTONIC=str(deadline), CUDA_VISIBLE_DEVICES='0')
        with (output / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(worker)], cwd=output, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
                result['status'] = 'complete' if process.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                result['status'] = 'timeout'
            finally:
                # Kill the whole group even if its leader already exited; setup can spawn descendants.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                result['returncode'] = process.returncode
    result['runtime_seconds'] = time.monotonic() - started
    result['deadline_exhausted'] = time.monotonic() >= deadline
    result['runner_source_sha256'] = hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest()
    stage = output / 'stage.json'
    if stage.exists():
        try:
            result['worker_stage'] = json.loads(stage.read_text())
        except (OSError, json.JSONDecodeError) as error:
            result['worker_stage_error'] = f'{type(error).__name__}: {error}'
    (output / 'execution.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    output = Path('/kaggle/working/tabpfn-pilot-execution')
    output.mkdir(exist_ok=False)
    result = supervise(output, WORKER_SOURCE, STARTED + TOTAL_BUDGET_SECONDS)
    result['total_runtime_seconds'] = time.monotonic() - STARTED
    (output / 'execution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
'''


def package():
    root = Path(__file__).resolve().parent
    output = root / 'kaggle-tabpfn-pilot'
    output.mkdir(exist_ok=False)
    runner = (root / 'train_tabpfn.py').read_text()
    worker = 'RUNNER_SOURCE = ' + repr(runner) + '\n' + WORKER
    script = SUPERVISOR.replace('TOTAL_BUDGET_SECONDS = 1710',
        'RUNNER_SOURCE = ' + repr(runner) + '\nWORKER_SOURCE = ' + repr(worker) + '\nTOTAL_BUDGET_SECONDS = 1710', 1)
    (output / 'train.py').write_text(script)
    py_compile.compile(str(output / 'train.py'), doraise=True)
    metadata = dict(id='wasidyms/s6e10-codex-tabpfn-fp32-pilot', title='S6E10 Codex TabPFN FP32 Pilot',
        code_file='train.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
        competition_sources=['playground-series-s6e10'], dataset_sources=[], kernel_sources=[], model_sources=[])
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(output)
    return output


if __name__ == '__main__':
    package()
