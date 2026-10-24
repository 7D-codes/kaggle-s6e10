"""Evaluate fixed equal weights. No label-fitted ensemble weights."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

p=argparse.ArgumentParser()
p.add_argument('--members',nargs='+',required=True,help='Paths to model-oof.csv files with matching folds')
p.add_argument('--output',required=True)
a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):
    raise FileExistsError('Use a fresh output directory')
oofs=[pd.read_csv(path) for path in a.members]
tests=[pd.read_csv(str(path).replace('-oof.csv','-predictions.csv')) for path in a.members]
base=oofs[0];test=tests[0]
member_path=Path(a.members[0])
member_metrics=json.loads((member_path.parent/'metrics.json').read_text())
member_model=member_path.name.removesuffix('-oof.csv')
member_model='equal_blend' if member_model=='blend' else member_model
split_seed=member_metrics.get('provenance',{}).get('split_seed',member_metrics[member_model].get('split_seed',42))
for o,t in zip(oofs,tests):
    assert o[['id','fold','target']].equals(base[['id','fold','target']])
    assert t.id.equals(test.id)
    assert np.isfinite(o.prediction).all() and o.prediction.between(0,1).all()
    assert np.isfinite(t.satisfaction).all() and t.satisfaction.between(0,1).all()
prob=np.mean([o.prediction.to_numpy() for o in oofs],axis=0)
pred=np.mean([t.satisfaction.to_numpy() for t in tests],axis=0)
meta={'equal_blend':{'split_seed':split_seed,'oof_auc':float(roc_auc_score(base.target,prob)),
    'fold_auc':[float(roc_auc_score(base.loc[base.fold==f,'target'],prob[base.fold==f])) for f in range(5)],
    'members':a.members,'weights':[1/len(oofs)]*len(oofs)},
    'correlation':np.corrcoef([o.prediction for o in oofs]).tolist(),
    'interpretation':'Development OOF; base models/feature sets selected on these data. No fitted weights.',
    'submissions_used':0}
base.assign(prediction=prob).to_csv(out/'blend-oof.csv',index=False)
test.assign(satisfaction=pred).to_csv(out/'blend-predictions.csv',index=False)
(out/'metrics.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
