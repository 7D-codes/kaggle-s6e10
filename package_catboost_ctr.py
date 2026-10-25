"""Build/compile the one frozen private CatBoost package; never launches it."""
import json
from pathlib import Path
import py_compile

from package_tabpfn_pilot import SUPERVISOR as BASE_SUPERVISOR

# Retain existing receipt and unconditional process-group cleanup; add only frozen caps/stages.
SUPERVISOR = BASE_SUPERVISOR.replace('TOTAL_BUDGET_SECONDS = 1500','TOTAL_BUDGET_SECONDS = 21600\nFIRST_FOLD_SECONDS = 3600')
SUPERVISOR = SUPERVISOR.replace('def supervise(output, worker_source, deadline):',
    'def supervise(output, worker_source, deadline, first_deadline=None):\n'
    '    first_deadline = STARTED + FIRST_FOLD_SECONDS if first_deadline is None else first_deadline')
SUPERVISOR = SUPERVISOR.replace("PILOT_DEADLINE_MONOTONIC=str(deadline), CUDA_VISIBLE_DEVICES='0'",
    "PILOT_DEADLINE_MONOTONIC=str(deadline), CATBOOST_STARTED_MONOTONIC=str(STARTED), CATBOOST_STAGE_PATH=str(output / 'stage.json'), CUDA_VISIBLE_DEVICES='0'")
SUPERVISOR = SUPERVISOR.replace('                process.wait(timeout=max(0, deadline - time.monotonic()))\n'
    "                result['status'] = 'complete' if process.returncode == 0 else 'failed'",'''                previous_stage = None
                while process.poll() is None:
                    now = time.monotonic()
                    limits = {'total': deadline}
                    if not (output / 'catboost/first-fold-complete.json').is_file():
                        limits['first_fold'] = first_deadline
                    stage_path = output / 'stage.json'
                    if stage_path.exists():
                        current_stage = json.loads(stage_path.read_text())
                        if current_stage.get('phase_deadline_monotonic') is not None:
                            limits['phase'] = float(current_stage['phase_deadline_monotonic'])
                        if current_stage != previous_stage:
                            print(json.dumps(current_stage), flush=True)
                            previous_stage = current_stage
                    kind = min(limits, key=limits.get)
                    if now >= limits[kind]:
                        result['deadline_kind'] = kind
                        break
                    try:
                        process.wait(timeout=min(1, max(.001, limits[kind] - now)))
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is not None:
                    result['status'] = 'complete' if process.returncode == 0 else 'failed' ''')
SUPERVISOR = SUPERVISOR.replace('tabpfn-pilot-execution','catboost-ctr-execution')

WORKER = '''import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path.cwd()
deadline = float(os.environ['PILOT_DEADLINE_MONOTONIC'])
def remaining():
    value = deadline-time.monotonic()
    if value <= 0: raise TimeoutError('CatBoost total budget exhausted')
    return value

def stage(name):
    path=root/'stage.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(stage=name,phase_deadline_monotonic=None)))
    temporary.replace(path)

stage('input_validation')
inputs=list(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv'))
if len(inputs)!=1: raise ValueError('Expected exactly one official competition train.csv')
for name,digest in INPUT_HASHES.items():
    with (inputs[0].parent/name).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Official input SHA256 mismatch')
teachers=list(Path('/kaggle/input').rglob('original-teacher/teacher-features.csv'))
producers=list(Path('/kaggle/input').rglob('reproducible-source/external_teacher.py'))
if len(teachers)!=1 or len(producers)!=1:raise ValueError('Expected exactly one qualified original teacher and producer attachment')
for name,digest in TEACHER_HASHES.items():
    with (teachers[0].parent/name).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Upstream teacher SHA256 mismatch')
with producers[0].open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=PRODUCER_SHA:raise ValueError('Upstream producer SHA256 mismatch')
stage('setup')
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--upgrade-strategy','only-if-needed',
    'catboost==1.2.8','pandas==2.3.3','numpy==2.1.3','scikit-learn==1.6.1'],check=True,timeout=remaining())
for name,source in SOURCES.items():(root/name).write_text(source)
(root/'source-hashes.json').write_text(json.dumps({name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCES.items()},indent=2))
stage('catboost')
subprocess.run([sys.executable,str(root/'train_catboost_ctr.py'),'--data',str(inputs[0].parent),
    '--teacher',str(teachers[0].parent),'--producer',str(producers[0]),'--output',str(root/'catboost')],check=True,timeout=remaining())
metrics=json.loads((root/'catboost/metrics.json').read_text())
if metrics['status']!='complete' or not metrics['full_oof'] or metrics['completed_folds']!=list(range(5)) or metrics['submissions']!=0:
    raise ValueError('Incomplete CatBoost artifact receipt')
stage('complete')
'''


def package(output=None):
    from train_catboost_ctr import INPUT_HASHES, TEACHER_HASHES, PRODUCER_SHA
    root=Path(__file__).resolve().parent
    output=Path(output) if output is not None else root/'kaggle-catboost-ctr'
    output.mkdir(parents=True,exist_ok=False)
    names=('train_catboost_ctr.py','train_tabpfn.py','train_kumo_pair.py','train_baselines.py','recipe_features.py','features.py')
    sources={name:(root/name).read_text() for name in names}
    worker='\n'.join(f'{name} = {value!r}' for name,value in dict(SOURCES=sources,INPUT_HASHES=INPUT_HASHES,TEACHER_HASHES=TEACHER_HASHES,PRODUCER_SHA=PRODUCER_SHA).items())+'\n'+WORKER
    script='RUNNER_SOURCE = '+repr(sources['train_catboost_ctr.py'])+'\nWORKER_SOURCE = '+repr(worker)+'\n'+SUPERVISOR
    (output/'train.py').write_text(script)
    py_compile.compile(str(output/'train.py'),doraise=True)
    metadata=dict(id='wasidyms/s6e10-codex-catboost-ctr-v1',title='S6E10 Codex CatBoost CTR v1',code_file='train.py',
        language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',
        competition_sources=['playground-series-s6e10'],dataset_sources=[],kernel_sources=['wasidyms/s6e10-batch6-route-recipe'],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    return output


if __name__=='__main__':print(package())
