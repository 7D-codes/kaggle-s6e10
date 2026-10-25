"""Build the frozen private paired Kumo package. Does not launch or submit."""
import json
from pathlib import Path
import py_compile

from package_tabpfn_pilot import SUPERVISOR as BASE_SUPERVISOR
from train_kumo_pair import CHECKPOINT_COMMIT, CHECKPOINT_SHA, INPUT_HASHES, SOURCE_COMMIT

# Reuse existing process-group cleanup/receipt handling; only the first-pair watchdog differs.
SUPERVISOR = BASE_SUPERVISOR.replace('TOTAL_BUDGET_SECONDS = 1500',
    'TOTAL_BUDGET_SECONDS = 28800\nFIRST_PAIR_SECONDS = 3600')
SUPERVISOR = SUPERVISOR.replace('def supervise(output, worker_source, deadline):',
    'def supervise(output, worker_source, deadline, first_deadline=None):\n'
    '    first_deadline = STARTED + FIRST_PAIR_SECONDS if first_deadline is None else first_deadline')
SUPERVISOR = SUPERVISOR.replace("PILOT_DEADLINE_MONOTONIC=str(deadline), CUDA_VISIBLE_DEVICES='0'",
    "PILOT_DEADLINE_MONOTONIC=str(deadline), KUMO_STARTED_MONOTONIC=str(STARTED), CUDA_VISIBLE_DEVICES='0'")
SUPERVISOR = SUPERVISOR.replace('                process.wait(timeout=max(0, deadline - time.monotonic()))\n'
    "                result['status'] = 'complete' if process.returncode == 0 else 'failed'",
    """                while process.poll() is None:
                    now = time.monotonic()
                    if now >= deadline:
                        result['deadline_kind'] = 'total'
                        break
                    if not (output / 'paired/first-fold-complete.json').is_file() and now >= first_deadline:
                        result['deadline_kind'] = 'first_pair'
                        break
                    try:
                        process.wait(timeout=min(1, max(0.001, deadline - now),
                            max(0.001, first_deadline - now) if not (output / 'paired/first-fold-complete.json').is_file() else 1))
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is not None:
                    result['status'] = 'complete' if process.returncode == 0 else 'failed'""")
SUPERVISOR = SUPERVISOR.replace('tabpfn-pilot-execution', 'kumo-pair-execution')

WORKER = '''import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path.cwd()
deadline = float(os.environ['PILOT_DEADLINE_MONOTONIC'])
def remaining():
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Paired total budget exhausted')
    return seconds

def stage(name):
    (root / 'stage.json').write_text(json.dumps({'stage': name, 'remaining_seconds': remaining()}))

stage('input_validation')
inputs = list(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv'))
if len(inputs) != 1:
    raise ValueError('Expected exactly one competition train.csv')
for name, digest in INPUT_HASHES.items():
    with (inputs[0].parent / name).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
            raise ValueError(f'Competition {name} SHA256 mismatch')
stage('setup')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
    '--upgrade-strategy', 'only-if-needed', 'torch==2.11.0', 'numpy==2.1.3', 'pandas==2.3.3',
    'scikit-learn==1.6.1', 'pyarrow==19.0.1', 'huggingface_hub==0.35.3', 'safetensors==0.6.2',
    'typing_extensions==4.15.0'], check=True, timeout=remaining())
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps',
    'git+https://github.com/NVIDIA/structured-data-models.git@' + SOURCE_COMMIT], check=True, timeout=remaining())
stage('checkpoint_download')
from huggingface_hub import HfApi, hf_hub_download
if HfApi().model_info('nvidia/Kumo-Tabular', revision='v1.0.1').sha != CHECKPOINT_COMMIT:
    raise ValueError('Official checkpoint tag resolved to unexpected commit')
checkpoint = Path(hf_hub_download('nvidia/Kumo-Tabular', 'small/classifier.pt', revision=CHECKPOINT_COMMIT))
with checkpoint.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != CHECKPOINT_SHA:
        raise ValueError('Official checkpoint SHA256 mismatch')
if checkpoint.stat().st_size != 109936978 or checkpoint.parent.parent.name != CHECKPOINT_COMMIT:
    raise ValueError('Official checkpoint size/snapshot mismatch')
(root / 'checkpoint-provenance.json').write_text(json.dumps(dict(repo='nvidia/Kumo-Tabular',
    tag='v1.0.1', commit=CHECKPOINT_COMMIT, sha256=CHECKPOINT_SHA, bytes=checkpoint.stat().st_size)))
stage('paired')
for name, source in SOURCES.items():
    (root / name).write_text(source)
environment = dict(os.environ, HF_HUB_OFFLINE='1')
subprocess.run([sys.executable, str(root / 'train_kumo_pair.py'), '--data', str(inputs[0].parent),
    '--checkpoint', str(checkpoint), '--output', str(root / 'paired')], check=True, timeout=remaining(), env=environment)
metrics = json.loads((root / 'paired/metrics.json').read_text())
if metrics['status'] != 'complete' or not metrics['full_oof'] or metrics['completed_folds'] != list(range(5)) or metrics['submissions'] != 0:
    raise ValueError('Incomplete or invalid paired receipt')
stage('complete')
'''


def package(output=None):
    root = Path(__file__).resolve().parent
    output = Path(output) if output is not None else root / 'kaggle-kumo-pair'
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: (root / name).read_text() for name in ('train_kumo_pair.py', 'train_tabpfn.py')}
    worker = '\n'.join(f'{name} = {value!r}' for name, value in dict(SOURCES=sources, INPUT_HASHES=INPUT_HASHES,
        SOURCE_COMMIT=SOURCE_COMMIT, CHECKPOINT_COMMIT=CHECKPOINT_COMMIT, CHECKPOINT_SHA=CHECKPOINT_SHA).items()) + '\n' + WORKER
    source = 'RUNNER_SOURCE = ' + repr(sources['train_kumo_pair.py']) + '\nWORKER_SOURCE = ' + repr(worker) + '\n' + SUPERVISOR
    (output / 'train.py').write_text(source)
    py_compile.compile(str(output / 'train.py'), doraise=True)
    metadata = dict(id='wasidyms/s6e10-codex-kumo-paired-v1', title='S6E10 Codex Kumo Paired v1',
        code_file='train.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
        competition_sources=['playground-series-s6e10'], dataset_sources=[], kernel_sources=[], model_sources=[])
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    return output


if __name__ == '__main__':
    print(package())
