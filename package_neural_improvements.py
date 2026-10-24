"""Package a three-arm neural ablation with real CUDA checkpoint acceptance. Never submits."""
import json
from pathlib import Path
import py_compile

root = Path(__file__).resolve().parent
out = root / 'kaggle-neural-improvements'
out.mkdir(exist_ok=True)
sources = {name: (root / name).read_text() for name in ('train_realmlp.py', 'predict_realmlp.py')}
script = '''import subprocess, sys, json, hashlib, gzip, gc
from pathlib import Path
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.1.3',
                'pandas==2.3.3', 'scikit-learn==1.6.1', 'pytabkit==1.7.3', 'cloudpickle==3.1.2'], check=True)
root = Path('/kaggle/working')
code = root / 'reproducible-source'
code.mkdir()
'''
script += 'sources = ' + repr(sources) + '\n'
script += "arms = " + repr([('control-three-epochs', 3, 'baseline'),
                            ('longer-six-epochs', 6, 'baseline'),
                            ('applicable-three-epochs', 3, 'applicability')]) + '\n'
script += '''for name, source in sources.items():
    (code / name).write_text(source)
sys.path.insert(0, str(code))
import numpy as np, pandas as pd, torch, cloudpickle
from importlib.metadata import version
from predict_realmlp import reconstruct_features
assert torch.cuda.is_available()
print('Actual CUDA:', torch.cuda.get_device_name(0), flush=True)
versions = {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'pytabkit', 'cloudpickle', 'torch')}
print('Runtime versions:', json.dumps(versions), flush=True)
data = next(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv')).parent
for name, expected in [('train.csv', '2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36'),
                       ('test.csv', 'ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6'),
                       ('sample_submission.csv', '6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71')]:
    with (data / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name

def smoke_inputs(data, root):
    small = root / 'smoke-data'
    small.mkdir()
    pd.read_csv(data / 'train.csv', nrows=4096).to_csv(small / 'train.csv', index=False)
    pd.read_csv(data / 'test.csv', nrows=64).to_csv(small / 'test.csv', index=False)
    pd.read_csv(data / 'sample_submission.csv', nrows=64).to_csv(small / 'sample_submission.csv', index=False)
    return small

def phase(name, epochs, features, data):
    subprocess.run([sys.executable, str(code / 'train_realmlp.py'), '--data', str(data),
                    '--output', str(root / name), '--seed', '42', '--epochs', str(epochs),
                    '--features', features], check=True)

def replay(name, data):
    run = root / name
    metadata = json.loads((run / 'metrics.json').read_text())
    train = pd.read_csv(data / 'train.csv')
    rows = pd.read_csv(data / 'test.csv', nrows=64)
    expected = pd.read_csv(run / 'realmlp-predictions.csv', nrows=64).satisfaction.to_numpy()
    actual = np.zeros(len(rows))
    for fold in range(5):
        prepared = reconstruct_features(train, rows, fold, seed=42,
                      features=metadata['provenance'].get('feature_mode', 'baseline'))
        with gzip.open(run / f'realmlp-fold{fold}.pkl.gz', 'rb') as stream:
            model = cloudpickle.load(stream)
        actual += model.predict_proba(prepared)[:, 1] / 5
        del model, prepared
        gc.collect()
        torch.cuda.empty_cache()
    assert np.ptp(expected) > 1e-6, 'Vacuous constant replay'
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    return {'rows': len(rows), 'max_abs_error': float(abs(actual - expected).max()),
            'rtol': 1e-6, 'atol': 1e-7, 'device': 'cuda', 'nonconstant': True}

small = smoke_inputs(data, root)
phase('cuda-applicability-smoke', 1, 'applicability', small)
smoke_acceptance = replay('cuda-applicability-smoke', small)
print('Actual generated-fixture CUDA raw-row reconstruction and checkpoint replay passed',
      json.dumps(smoke_acceptance), flush=True)
acceptance = {}
for name, epochs, features in arms:
    phase(name, epochs, features, data)
    acceptance[name] = replay(name, data)
    print('Full-arm CUDA replay passed', name, json.dumps(acceptance[name]), flush=True)
summary = {'seed': 42, 'folds': 5, 'arms': acceptance, 'versions': versions,
           'smoke': smoke_acceptance,
           'validation': 'Fixed-epoch development OOF, reused rows; not untouched holdout',
           'submissions': 0}
(root / 'neural-improvements-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
'''
(out / 'train.py').write_text(script)
py_compile.compile(str(out / 'train.py'), doraise=True)
metadata = {'id': 'wasidyms/s6e10-batch7-neural-improvements',
            'title': 'S6E10 Batch7 Neural Improvements', 'code_file': 'train.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
            'competition_sources': ['playground-series-s6e10'],
            'dataset_sources': [], 'kernel_sources': [], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(out)
