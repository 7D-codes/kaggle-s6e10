from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT/'data/train.csv')
test = pd.read_csv(ROOT/'data/test.csv')
y = train.satisfaction.astype(int).to_numpy()
expected_by_seed = {}

def expected_folds(seed):
    if seed not in expected_by_seed:
        expected = np.full(len(y), -1)
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(train,y)):
            assert not np.intersect1d(a,b).size
            assert (expected[b] == -1).all()
            expected[b] = fold
        assert (expected >= 0).all()
        expected_by_seed[seed] = expected
    return expected_by_seed[seed]

checked = 0
for metrics_file in (ROOT/'results').rglob('metrics.json'):
    metrics = json.loads(metrics_file.read_text())
    for model in ('lightgbm','catboost','xgboost','realmlp','equal_blend'):
        stem = 'blend' if model == 'equal_blend' else model
        oof_file = metrics_file.parent/f'{stem}-oof.csv'
        if model not in metrics or not oof_file.exists():
            continue
        oof = pd.read_csv(oof_file)
        pred = pd.read_csv(metrics_file.parent/f'{stem}-predictions.csv')
        assert oof.id.equals(train.id) and pred.id.equals(test.id)
        seed = metrics.get('provenance',{}).get('split_seed',metrics[model].get('split_seed',42))
        assert np.array_equal(oof.fold,expected_folds(seed))
        assert np.array_equal(oof.target,y)
        assert np.isfinite(oof.prediction).all() and np.isfinite(pred.satisfaction).all()
        assert oof.prediction.between(0,1).all() and pred.satisfaction.between(0,1).all()
        score = roc_auc_score(y,oof.prediction)
        assert abs(score-metrics[model]['oof_auc']) < 1e-10
        print(f'{metrics_file.parent.name}/{model}: independently recomputed AUC {score:.8f}')
        checked += 1
assert checked, 'No completed models available to verify'
print(f'PASS: {checked} model outputs verified; every row held out exactly once')
