"""Package a read-only GPU/CPU checkpoint replay audit. Never trains or submits."""
from pathlib import Path
import json
import py_compile

root = Path(__file__).resolve().parent
out = root / 'kaggle-route-replay'
out.mkdir(exist_ok=True)
sources = {name: (root / name).read_text() for name in
           ('features.py', 'recipe_features.py', 'predict_recipe.py')}
manifest = json.loads((root / 'results/batch6-route-recipe/route-full-xgboost/inference-manifest.json').read_text())
script = '''import subprocess, sys, json, hashlib, platform
from pathlib import Path
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost==3.2.0', 'scikit-learn==1.9.1', 'pandas==3.1.0', 'pyarrow'], check=True)
root = Path('/kaggle/working')
code = root / 'audit-source'
code.mkdir()
'''
script += 'sources = ' + repr(sources) + '\nmanifest = ' + repr(manifest) + '\n'
script += '''for name, source in sources.items():
    (code / name).write_text(source)
sys.path.insert(0, str(code))
import numpy as np, pandas as pd, xgboost as xgb, sklearn, joblib
from recipe_features import transform_recipe, BASE
from predict_recipe import _teacher_probabilities
inputs = Path('/kaggle/input')
run = next(inputs.rglob('route-full-xgboost/xgboost-fold0.json')).parent
teacher = run.parent / 'original-teacher'
# Bind to the exact downloaded v1 artifacts before deserializing fitted state.
for directory, hashes in ((run, manifest['run_hashes']), (teacher, manifest['teacher_hashes'])):
    for name, expected in hashes.items():
        with (directory / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, str(directory / name)
data = next(inputs.rglob('playground-series-s6e10/test.csv'))
raw = pd.read_csv(data, nrows=128)
expected = pd.read_csv(run / 'xgboost-predictions.csv', nrows=len(raw)).satisfaction.to_numpy()
cached = pd.read_csv(teacher / 'teacher-features.csv', usecols=['id', 'teacher_original_xgb']).set_index('id').loc[raw.id, 'teacher_original_xgb'].to_numpy()
source = json.loads((teacher / 'provenance.json').read_text())
frame = raw[source['features']].copy()
for column, levels in source['categorical_levels'].items():
    frame[column] = pd.Categorical(frame[column].where(frame[column].isin(levels)), categories=levels)
teacher_scores = {'cached': cached, 'reconstructed_cpu': _teacher_probabilities(raw[BASE], teacher)}
tm = xgb.XGBClassifier(device='cuda', enable_categorical=True)
tm.load_model(teacher / 'teacher-model.json')
tm.set_params(device='cuda')
teacher_scores['reconstructed_cuda'] = tm.predict_proba(frame)[:, 1]
assert json.loads(tm.get_booster().save_config())['learner']['generic_param']['device'].startswith('cuda')
predictions = {device + '_' + variant: np.zeros(len(raw)) for device in ('cpu', 'cuda') for variant in teacher_scores}
for fold in range(5):
    state = joblib.load(run / f'xgboost-preprocessor-fold{fold}.joblib')
    prepared = transform_recipe(raw[BASE], state)
    prepared.to_parquet(root / f'prepared-fold{fold}.parquet', index=False)
    for device in ('cpu', 'cuda'):
        model = xgb.XGBClassifier(device=device, enable_categorical=True)
        model.load_model(run / f'xgboost-fold{fold}.json')
        model.set_params(device=device)
        for variant, scores in teacher_scores.items():
            prepared['teacher_original_xgb'] = scores
            assert list(prepared) == manifest['features']
            predictions[device + '_' + variant] += model.predict_proba(prepared)[:, 1] / 5
        actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        assert actual.startswith(device), (device, actual)
    print(json.dumps({'fold_complete': fold}), flush=True)
output = pd.DataFrame({'id': raw.id, 'expected': expected, **predictions})
output.to_csv(root / 'replay-predictions.csv', index=False)
report = {
    'rows': len(raw), 'tolerance': 1e-7, 'training_runs': 0, 'submissions': 0,
    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgb.__version__},
    'teacher_max_abs_errors': {key: float(np.max(np.abs(value - cached))) for key, value in teacher_scores.items()},
    'replay': {key: {'max_abs_error': float(np.max(np.abs(value - expected))), 'rows_outside_tolerance': int(np.sum(np.abs(value - expected) > 1e-7))} for key, value in predictions.items()},
    'upstream': 'wasidyms/s6e10-batch6-route-recipe version1, all native model/state/teacher hashes verified',
}
(root / 'replay-audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
'''
(out / 'audit.py').write_text(script)
py_compile.compile(str(out / 'audit.py'), doraise=True)
metadata = {
    'id': 'wasidyms/s6e10-batch6-checkpoint-replay-audit', 'title': 'S6E10 Batch6 Checkpoint Replay Audit',
    'code_file': 'audit.py', 'language': 'python', 'kernel_type': 'script',
    'is_private': True, 'enable_gpu': True, 'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4', 'competition_sources': ['playground-series-s6e10'],
    'dataset_sources': [], 'kernel_sources': ['wasidyms/s6e10-batch6-route-recipe'], 'model_sources': [],
}
(out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(out)
