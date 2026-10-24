import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize('seed,legacy', [(2026, False), (42, True)])
def test_nested_equal_blends_preserve_member_split_seed(tmp_path, seed, legacy):
    script = Path(__file__).with_name('blend_equal.py')
    members = []
    for model in ('lightgbm', 'xgboost'):
        directory = tmp_path / model
        directory.mkdir()
        path = directory / f'{model}-oof.csv'
        pd.DataFrame({'id': range(10), 'fold': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                      'target': [0, 1] * 5, 'prediction': [0.2, 0.8] * 5}).to_csv(path, index=False)
        pd.DataFrame({'id': [10, 11], 'satisfaction': [0.3, 0.7]}).to_csv(
            directory / f'{model}-predictions.csv', index=False)
        metrics = {model: {}}
        if not legacy:
            metrics['provenance'] = {'split_seed': seed}
        (directory / 'metrics.json').write_text(json.dumps(metrics))
        members.append(str(path))
    first = tmp_path / 'trees'
    second = tmp_path / 'nested'
    for paths, output in ((members, first), ([str(first / 'blend-oof.csv'), members[0]], second)):
        result = subprocess.run([sys.executable, str(script), '--members', *paths,
                                 '--output', str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        metrics = json.loads((output / 'metrics.json').read_text())
        assert metrics['equal_blend']['split_seed'] == seed
