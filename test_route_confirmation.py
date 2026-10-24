"""Exercise the generated remote smoke's exact input setup locally."""
import ast
import runpy
import shutil
from argparse import Namespace
from pathlib import Path

import pandas as pd

from test_predict_recipe import make_data
from train_baselines import run


def test_generated_smoke_inputs_run_shared_trainer(tmp_path):
    project = Path(__file__).resolve().parent
    for name in ('package_route_confirmation.py', 'features.py', 'external_teacher.py',
                 'recipe_features.py', 'train_baselines.py', 'predict_recipe.py'):
        shutil.copy2(project / name, tmp_path / name)
    manifest = Path('results/batch6-route-recipe/route-full-xgboost/inference-manifest.json')
    (tmp_path / manifest).parent.mkdir(parents=True)
    shutil.copy2(project / manifest, tmp_path / manifest)
    runpy.run_path(str(tmp_path / 'package_route_confirmation.py'))
    generated = tmp_path / 'kaggle-route-confirmation/train.py'
    tree = ast.parse(generated.read_text())
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets)
    start = next(i for i, node in enumerate(tree.body) if assignment(node, 'small'))
    stop = next(i for i, node in enumerate(tree.body) if assignment(node, 'smoke_teacher'))
    train, test = make_data(tmp_path / 'data')
    namespace = {'root': tmp_path, 'data': tmp_path / 'data', 'train': train, 'test': test,
                 'source': pd.read_csv(tmp_path / 'data/original.csv'), 'pd': pd}
    exec(compile(ast.Module(body=tree.body[start:stop], type_ignores=[]), str(generated), 'exec'), namespace)
    small = namespace['small']
    sample = pd.read_csv(small / 'sample_submission.csv')
    assert sample.id.equals(test.id.iloc[:64].reset_index(drop=True))
    assert list(sample) == ['id', 'satisfaction']
    output = tmp_path / 'native-smoke'
    run(Namespace(data=str(small), output=str(output), original=str(small / 'original.csv'),
                  features='recipe_full', preset='route', models='xgboost', rounds=3,
                  seed=2026, leaves=7, gpu=False, teacher_csv=None, auxiliary_csv=None))
    assert len(pd.read_csv(output / 'xgboost-predictions.csv')) == len(sample)
