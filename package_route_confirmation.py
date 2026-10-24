"""Package three bounded seed2026 robustness arms. Never submits."""
from pathlib import Path
import json
import py_compile

root = Path(__file__).resolve().parent
out = root / 'kaggle-route-confirmation'
out.mkdir(exist_ok=True)
sources = {name: (root / name).read_text() for name in
           ('features.py', 'external_teacher.py', 'recipe_features.py', 'train_baselines.py', 'predict_recipe.py')}
manifest = json.loads((root / 'results/batch6-route-recipe/route-full-xgboost/inference-manifest.json').read_text())
script = '''import subprocess, sys, json, shutil, hashlib
from pathlib import Path
subprocess.run(['nvidia-smi'], check=True)
root = Path('/kaggle/working')
code = root / 'reproducible-source'
code.mkdir()
'''
script += 'sources = ' + repr(sources) + '\nupstream_manifest = ' + repr(manifest) + '\n'
script += '''for name, source in sources.items():
    (code / name).write_text(source)
sys.path.insert(0, str(code))
inputs = Path('/kaggle/input')
data = next(inputs.rglob('playground-series-s6e10/train.csv')).parent
original = next(inputs.rglob('binary-aviation-satisfaction-129k/data.csv'))
upstream = next(inputs.rglob('route-full-xgboost/xgboost-fold0.json')).parent.parent
for path, expected in ((data / 'train.csv', '2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36'),
                       (data / 'test.csv', 'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'),
                       (original, '2ad76465f6115f93d16121891a6b1235acb8007a4ab065cae0dc46b60ad116e3')):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, str(path)
teacher = root / 'original-teacher'
teacher.mkdir()
for name, expected in upstream_manifest['teacher_hashes'].items():
    path = upstream / 'original-teacher' / name
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, str(path)
    shutil.copy2(path, teacher / name)

def install(packages):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)

def phase(name, model, features, preset, rounds, use_teacher=False):
    command = [sys.executable, str(code / 'train_baselines.py'), '--data', str(data),
               '--original', str(original), '--output', str(root / name), '--models', model,
               '--features', features, '--preset', preset, '--rounds', str(rounds), '--seed', '2026']
    if model == 'xgboost':
        command += ['--gpu']
    if use_teacher:
        command += ['--teacher-csv', str(teacher / 'teacher-features.csv')]
    subprocess.run(command, check=True)

# Match the historical incumbent XGBoost component, not the new route preset.
# Its original source and logs establish rich120features,3500rounds,XGB3.4.1.
install(['numpy==2.1.3', 'pandas==2.3.3', 'scikit-learn==1.6.1', 'xgboost==3.4.1'])
subprocess.run([sys.executable, '-c', "import xgboost,pandas,sklearn;assert xgboost.__version__=='3.4.1';assert pandas.__version__=='2.3.3';assert sklearn.__version__=='1.6.1';print('Historical XGBoost runtime versions verified',flush=True)"], check=True)
subprocess.run([sys.executable, '-c', "import xgboost as xgb,json; m=xgb.XGBClassifier(n_estimators=2,device='cuda',tree_method='hist');m.fit([[0],[1],[2],[3]],[0,0,1,1]);actual=json.loads(m.get_booster().save_config())['learner']['generic_param']['device'];assert actual.startswith('cuda'),actual;print('Historical control actual CUDA smoke passed',flush=True)"], check=True)
phase('incumbent-xgboost-seed2026', 'xgboost', 'rich', 'baseline', 3500)

# New full-recipe runtime matches batch6, adding frozen rarity lookups.
install(['numpy==2.1.3', 'pandas==3.1.0', 'scikit-learn==1.9.1', 'xgboost==3.2.0', 'lightgbm==4.7.0', 'pyarrow'])
import numpy as np, pandas as pd, joblib
from external_teacher import exclude_competition_overlaps
from predict_recipe import seal_recipe_bundle, predict_recipe
from recipe_features import transform_recipe
train = pd.read_csv(data / 'train.csv')
test = pd.read_csv(data / 'test.csv')
source, removed = exclude_competition_overlaps(pd.read_csv(original), train, test)
assert removed == 21
small = root / 'smoke-data'
small.mkdir()
train.iloc[:512].to_csv(small / 'train.csv', index=False)
test.iloc[:64].to_csv(small / 'test.csv', index=False)
pd.read_csv(data / 'sample_submission.csv', nrows=64).to_csv(small / 'sample_submission.csv', index=False)
source.to_csv(small / 'original.csv', index=False)
smoke_teacher = root / 'smoke-teacher'
smoke_teacher.mkdir()
for name in ('teacher-model.json', 'provenance.json'):
    shutil.copy2(teacher / name, smoke_teacher / name)
cache = pd.read_csv(teacher / 'teacher-features.csv').set_index('id')
ids = pd.concat([train.id.iloc[:512], test.id.iloc[:64]], ignore_index=True)
cache.loc[ids].reset_index().to_csv(smoke_teacher / 'teacher-features.csv', index=False)
smoke = root / 'smoke-xgboost'
subprocess.run([sys.executable, str(code / 'train_baselines.py'), '--data', str(small),
               '--original', str(small / 'original.csv'), '--output', str(smoke), '--models', 'xgboost',
               '--features', 'recipe_full', '--preset', 'route', '--rounds', '3', '--gpu', '--seed', '2026',
               '--teacher-csv', str(smoke_teacher / 'teacher-features.csv')], check=True)
seal_recipe_bundle(smoke, smoke_teacher, model='xgboost')
raw = test.iloc[:64].drop(columns='id')
expected = pd.read_csv(smoke / 'xgboost-predictions.csv').satisfaction.to_numpy()
assert np.ptp(expected) > 1e-6
np.testing.assert_allclose(predict_recipe(raw, smoke, smoke_teacher, model='xgboost', device='cuda'), expected, rtol=0, atol=1e-7)
state = joblib.load(smoke / 'xgboost-preprocessor-fold0.joblib')
assert 'rarity_maps' in state
transform_recipe(raw, state).to_parquet(root / 'portable-smoke-features.parquet', index=False)
print('Nonconstant CUDA raw-row native checkpoint replay with frozen rarity state passed', flush=True)
del train, test, source, cache

phase('route-full-seed2026', 'lightgbm', 'recipe_full', 'route', 8000, True)
phase('route-full-xgboost-seed2026', 'xgboost', 'recipe_full', 'route', 12000, True)
acceptance = {}
for name, model in (('route-full-seed2026', 'lightgbm'), ('route-full-xgboost-seed2026', 'xgboost')):
    run = root / name
    seal_recipe_bundle(run, teacher, model=model)
    expected = pd.read_csv(run / f'{model}-predictions.csv', nrows=64).satisfaction.to_numpy()
    actual = predict_recipe(raw, run, teacher, model=model, device='cuda')
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    acceptance[name] = {'rows': 64, 'raw_row_cuda_replay_max_abs_error': float(np.max(abs(actual - expected)))}
report = {'seed': 2026, 'folds': 5, 'arms': acceptance, 'baseline_control': 'incumbent-xgboost-seed2026',
          'recipe_change': 'Frozen training-derived rarity lookup outputs; same mathematical features and parameters',
          'validation': 'Development robustness check on reused rows, not untouched holdout', 'submissions': 0}
(root / 'confirmation-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
'''
(out / 'train.py').write_text(script)
py_compile.compile(str(out / 'train.py'), doraise=True)
metadata = {
    'id': 'wasidyms/s6e10-batch6-route-confirmation', 'title': 'S6E10 Batch6 Route Confirmation',
    'code_file': 'train.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
    'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
    'competition_sources': ['playground-series-s6e10'],
    'dataset_sources': ['arseniyshutko/binary-aviation-satisfaction-129k'],
    'kernel_sources': ['wasidyms/s6e10-batch6-route-recipe'], 'model_sources': [],
}
(out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(out)
