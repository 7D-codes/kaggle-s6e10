"""Evaluate a predeclared equal-weight blend, without weight fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parent
lgb = root/'results/batch1-lightgbm'
cat = root/'results/batch1-catboost'
a = pd.read_csv(lgb/'lightgbm-oof.csv')
b = pd.read_csv(cat/'catboost-oof.csv')
ta = pd.read_csv(lgb/'lightgbm-predictions.csv')
tb = pd.read_csv(cat/'catboost-predictions.csv')
assert a.id.equals(b.id) and a.fold.equals(b.fold) and a.target.equals(b.target)
assert ta.id.equals(tb.id)
p = (a.prediction+b.prediction)/2
t = (ta.satisfaction+tb.satisfaction)/2
assert np.isfinite(p).all() and t.between(0,1).all()
metrics = {name: json.loads((folder/'metrics.json').read_text())[name] for name,folder in [('lightgbm',lgb),('catboost',cat)]}
metrics['equal_blend'] = {'oof_auc':float(roc_auc_score(a.target,p)),
    'fold_auc':[float(roc_auc_score(a.loc[a.fold==f,'target'],p[a.fold==f])) for f in range(5)],
    'weights':{'lightgbm':0.5,'catboost':0.5}}
metrics['prediction_correlation']=float(a.prediction.corr(b.prediction))
metrics['interpretation']='Out-of-fold development estimates, not Kaggle leaderboard scores. Early stopping uses each validation fold; final selection needs independent confirmation.'
metrics['submissions_used']=0
out=root/'results/batch1-blend'
out.mkdir(exist_ok=True)
pd.DataFrame({'id':a.id,'fold':a.fold,'target':a.target,'prediction':p}).to_csv(out/'blend-oof.csv',index=False)
pd.DataFrame({'id':ta.id,'satisfaction':t}).to_csv(out/'blend-predictions.csv',index=False)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
