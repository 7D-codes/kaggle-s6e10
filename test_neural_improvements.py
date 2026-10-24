"""Exercise the exact generated smoke fixtures before allocating a remote GPU."""
import ast
import runpy
import shutil
from pathlib import Path

import pandas as pd

from test_predict_recipe import make_data
from train_realmlp import prepare


def test_generated_neural_smoke_inputs_are_complete(tmp_path):
    project = Path(__file__).resolve().parent
    for name in ('package_neural_improvements.py', 'train_realmlp.py', 'predict_realmlp.py'):
        shutil.copy2(project / name, tmp_path / name)
    runpy.run_path(str(tmp_path / 'package_neural_improvements.py'))
    generated = tmp_path / 'kaggle-neural-improvements/train.py'
    tree = ast.parse(generated.read_text())
    fixture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'smoke_inputs')
    namespace = {'pd': pd, 'Path': Path}
    exec(compile(ast.Module(body=[fixture], type_ignores=[]), str(generated), 'exec'), namespace)
    train, test = make_data(tmp_path / 'data')
    small = namespace['smoke_inputs'](tmp_path / 'data', tmp_path)
    a = pd.read_csv(small / 'train.csv')
    b = pd.read_csv(small / 'test.csv')
    sample = pd.read_csv(small / 'sample_submission.csv')
    assert a.id.equals(train.id.iloc[:4096].reset_index(drop=True))
    assert b.id.equals(test.id.iloc[:64].reset_index(drop=True))
    assert b.id.equals(sample.id)
    assert list(sample) == ['id', 'satisfaction']
    x = a.drop(columns=['id', 'satisfaction'])
    t = b.drop(columns='id')
    ready = prepare(x.iloc[:100], x.iloc[100:], t, a.satisfaction.iloc[:100].astype(int), 42)
    assert len(ready[2]) == len(sample)
    assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in ('sources', 'arms')}
    assert assignments['arms'] == [('control-three-epochs', 3, 'baseline'),
                                    ('longer-six-epochs', 6, 'baseline'),
                                    ('applicable-three-epochs', 3, 'applicability')]
    for name, source in assignments['sources'].items():
        assert source == (tmp_path / name).read_text()
    import json
    metadata = json.loads((generated.parent / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['competition_sources'] == ['playground-series-s6e10']
