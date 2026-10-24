"""Reproducible five-fold baselines. Never submits to the competition."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load_teacher_features(path,train_ids,test_ids):
    frame=pd.read_csv(path)
    if list(frame)!=['id','teacher_original_xgb'] or not frame.id.equals(pd.concat([train_ids,test_ids],ignore_index=True)):
        raise ValueError('Teacher features must match all train then test IDs exactly')
    prob=frame.teacher_original_xgb.to_numpy(dtype=float)
    if not np.isfinite(prob).all() or not ((prob>=0)&(prob<=1)).all():
        raise ValueError('Invalid teacher probabilities')
    return prob[:len(train_ids)],prob[len(train_ids):]


def load_auxiliary_features(path,train_ids,test_ids):
    frame=pd.read_csv(path)
    if len(frame.columns)<2 or frame.columns[0]!='id' or not frame.id.equals(pd.concat([train_ids,test_ids],ignore_index=True)):
        raise ValueError('Auxiliary rows must match all train then test IDs exactly')
    features=frame.drop(columns='id')
    if not all(c.startswith('aux_expected_') for c in features):
        raise ValueError('Only auxiliary expected-rating columns are accepted')
    values=features.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values>=0)&(values<=5)).all():
        raise ValueError('Auxiliary expectations must be finite in [0,5]')
    return features.iloc[:len(train_ids)].reset_index(drop=True),features.iloc[len(train_ids):].reset_index(drop=True)


def model_parameters(name,preset,rounds,leaves,seed,gpu):
    if name=='lightgbm':
        p=dict(n_estimators=rounds,learning_rate=0.04,num_leaves=leaves,min_child_samples=50,
            colsample_bytree=0.85,subsample=0.85,subsample_freq=1,reg_lambda=1.0,
            n_jobs=6,random_state=seed,verbosity=-1)
        if preset=='route':
            p.update(learning_rate=0.02,max_depth=8,num_leaves=127,colsample_bytree=0.5,
                subsample=0.9,min_child_samples=20,reg_lambda=5.0,reg_alpha=0.05)
    elif name=='xgboost':
        p=dict(n_estimators=rounds,learning_rate=0.03,max_depth=7,min_child_weight=5,
            subsample=0.9,colsample_bytree=0.7,reg_lambda=5.0,reg_alpha=0.05,tree_method='hist',
            device='cuda' if gpu else 'cpu',enable_categorical=True,max_cat_to_onehot=8,
            eval_metric='auc',early_stopping_rounds=150,random_state=seed,n_jobs=6)
        if preset=='route':
            p.update(learning_rate=0.02,max_depth=8,colsample_bytree=0.5,early_stopping_rounds=300)
    else:
        raise ValueError('Parameter presets support LightGBM and XGBoost only')
    if preset not in ('baseline','route'):
        raise ValueError('Unknown model preset')
    return p


def run(args):
    root = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists() or any(out.glob('*-fold*')) or any(out.glob('*-oof.csv')) or any(out.glob('*-predictions.csv')):
        raise FileExistsError('Use a fresh output directory to preserve previous results')
    tr, te = pd.read_csv(root/'train.csv'), pd.read_csv(root/'test.csv')
    sample = pd.read_csv(root/'sample_submission.csv')
    assert tr.id.is_unique and te.id.is_unique and sample.id.equals(te.id)
    assert not tr.satisfaction.isna().any()
    y = tr.satisfaction.astype(int).to_numpy()
    X = tr.drop(columns=['id', 'satisfaction'])
    T = te.drop(columns=['id'])
    assert list(X.columns) == list(T.columns)
    teacher_path=getattr(args,'teacher_csv',None)
    teacher_train,teacher_test=(None,None) if teacher_path is None else load_teacher_features(teacher_path,tr.id,te.id)
    auxiliary_path=getattr(args,'auxiliary_csv',None)
    auxiliary_train,auxiliary_test=(None,None) if auxiliary_path is None else load_auxiliary_features(auxiliary_path,tr.id,te.id)
    cats = X.select_dtypes(include=['object', 'string']).columns.tolist()
    seed = getattr(args, 'seed', 42)
    feature_kind = getattr(args, 'features', 'none')
    preset=getattr(args,'preset','baseline')
    if preset=='route' and any(n not in ('lightgbm','xgboost') for n in args.models.split(',')):
        raise ValueError('Route preset supports LightGBM and XGBoost only')
    original = None
    if feature_kind != 'none':
        from features import encode_original_target
        original = pd.read_csv(args.original)
        original['satisfaction'] = encode_original_target(original.satisfaction)
        # Remove original rows that exactly match ANY competition row, without labels.
        comp_keys = pd.util.hash_pandas_object(pd.concat([X,T],ignore_index=True),index=False)
        orig_keys = pd.util.hash_pandas_object(original[list(X)],index=False)
        keep = ~orig_keys.isin(comp_keys)
        print(json.dumps({'original_rows':len(original),'original_exact_overlap_removed':int((~keep).sum())}),flush=True)
        original = original.loc[keep].reset_index(drop=True)
        assert len(original)>0 and original.satisfaction.nunique()==2
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X,y))
    fold_ids = np.full(len(y), -1)
    for f, (_, va) in enumerate(splits):
        fold_ids[va] = f
    results = {}
    oofs, tests = [], []
    for name in args.models.split(','):
        started = time.time()
        oof, pred, scores, iterations = np.zeros(len(y)), np.zeros(len(T)), [], []
        for f, (a, b) in enumerate(splits):
            xa, xb, xt = X.iloc[a].copy(), X.iloc[b].copy(), T.copy()
            if feature_kind in ('recipe_full','recipe_te_only'):
                from recipe_features import prepare_recipe
                import joblib
                xa,xb,xt,preprocessor=prepare_recipe(xa,xb,xt,y[a],original,seed+f,
                    mode=feature_kind.removeprefix('recipe_'))
                joblib.dump(preprocessor,out/f'{name}-preprocessor-fold{f}.joblib',compress=3)
            elif feature_kind != 'none':
                from features import add_features
                xa, xb, xt = add_features(xa, xb, xt, original, feature_kind)
            if teacher_path is not None:
                xa['teacher_original_xgb']=teacher_train[a]
                xb['teacher_original_xgb']=teacher_train[b]
                xt['teacher_original_xgb']=teacher_test
            if auxiliary_path is not None:
                xa[auxiliary_train.columns]=auxiliary_train.iloc[a].to_numpy()
                xb[auxiliary_train.columns]=auxiliary_train.iloc[b].to_numpy()
                xt[auxiliary_train.columns]=auxiliary_test.to_numpy()
            fold_cats = xa.select_dtypes(include=['object','string','category']).columns.tolist()
            if name == 'lightgbm':
                import lightgbm as lgb
                for c in fold_cats:
                    # Categories are learned exclusively from this training fold.
                    levels = pd.Index(xa[c].dropna().unique())
                    for frame in (xa, xb, xt):
                        frame[c] = pd.Categorical(frame[c], categories=levels)
                model = lgb.LGBMClassifier(**model_parameters(name,preset,args.rounds,
                    getattr(args,'leaves',63),seed+f,args.gpu))
                model.fit(xa, y[a], eval_set=[(xb,y[b])], eval_metric='auc',
                    callbacks=[lgb.early_stopping(200 if preset=='route' else 100, verbose=False)])
                best = int(model.best_iteration_)
                model.booster_.save_model(str(out/f'{name}-fold{f}.txt'))
            elif name == 'catboost':
                from catboost import CatBoostClassifier
                for frame in (xa, xb, xt):
                    for c in fold_cats:
                        frame[c] = frame[c].fillna('__MISSING__').astype(str)
                    # Fixed sentinel, no statistics learned from validation/test.
                    frame.fillna(-999, inplace=True)
                model = CatBoostClassifier(iterations=args.rounds, learning_rate=0.05,
                    depth=7, loss_function='Logloss', eval_metric='AUC',
                    task_type='GPU' if args.gpu else 'CPU', thread_count=6,
                    random_seed=seed+f, verbose=False, allow_writing_files=False)
                model.fit(xa,y[a],cat_features=fold_cats,eval_set=(xb,y[b]),early_stopping_rounds=100)
                best = int(model.get_best_iteration()+1)
                model.save_model(str(out/f'{name}-fold{f}.cbm'))
            elif name == 'xgboost':
                import xgboost as xgb
                for c in fold_cats:
                    levels = pd.Index(xa[c].dropna().unique())
                    for frame in (xa,xb,xt):
                        frame[c] = pd.Categorical(frame[c],categories=levels)
                model = xgb.XGBClassifier(**model_parameters(name,preset,args.rounds,
                    getattr(args,'leaves',63),seed+f,args.gpu))
                model.fit(xa,y[a],eval_set=[(xb,y[b])],verbose=False)
                if args.gpu:
                    actual_device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
                    assert actual_device.startswith('cuda'), f'XGBoost fell back to {actual_device}'
                best = int(model.best_iteration+1)
                model.save_model(out/f'{name}-fold{f}.json')
            else:
                raise ValueError(name)
            oof[b] = model.predict_proba(xb)[:,1]
            pred += model.predict_proba(xt)[:,1]/5
            score = float(roc_auc_score(y[b],oof[b]))
            scores.append(score)
            iterations.append(best)
            print(json.dumps({'model':name,'fold':f,'auc':score,'iterations':best}),flush=True)
        assert np.isfinite(oof).all() and np.isfinite(pred).all()
        pd.DataFrame({'id':tr.id,'fold':fold_ids,'target':y,'prediction':oof}).to_csv(out/f'{name}-oof.csv',index=False)
        pd.DataFrame({'id':te.id,'satisfaction':pred}).to_csv(out/f'{name}-predictions.csv',index=False)
        results[name] = {'oof_auc':float(roc_auc_score(y,oof)), 'fold_auc':scores,
            'fold_std':float(np.std(scores)), 'best_iterations':iterations,'seconds':time.time()-started,
            'gpu':bool(args.gpu and name in ('catboost','xgboost')),
            'feature_kind':feature_kind,'feature_count':len(xa.columns),
            'leaves':model.get_params().get('num_leaves',getattr(args,'leaves',63)),
            'split_seed':seed,'model_preset':preset}
        if name in ('lightgbm','xgboost'):
            results[name]['parameters']=model_parameters(name,preset,args.rounds,getattr(args,'leaves',63),seed,args.gpu)
        oofs.append(oof)
        tests.append(pred)
        (out/'metrics.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({name:results[name]}),flush=True)
    if len(oofs)>1:
        avg=np.mean(oofs,axis=0)
        results['equal_blend']={'oof_auc':float(roc_auc_score(y,avg))}
        pd.DataFrame({'id':te.id,'satisfaction':np.mean(tests,axis=0)}).to_csv(out/'blend-predictions.csv',index=False)
    import sklearn
    from importlib.metadata import version
    import hashlib
    results['provenance']={'folds':5,'split_seed':seed,'features':list(xa.columns),
        'train_rows':len(X),'test_rows':len(T),
        'model_versions':{name:version(name) for name in args.models.split(',')},
        'arguments':vars(args),'validation':'early-stopped development OOF, not an untouched holdout',
        'pandas_version':pd.__version__,'sklearn_version':sklearn.__version__,'submissions':0}
    if teacher_path is not None:
        results['provenance']['teacher_csv_sha256']=hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
    if auxiliary_path is not None:
        results['provenance']['auxiliary_csv_sha256']=hashlib.sha256(Path(auxiliary_path).read_bytes()).hexdigest()
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/kaggle/input/playground-series-s6e10')
    p.add_argument('--output',default='/kaggle/working')
    p.add_argument('--models',default='lightgbm,catboost')
    p.add_argument('--rounds',type=int,default=2000)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--features',choices=['none','original','rich','recipe_full','recipe_te_only'],default='none')
    p.add_argument('--preset',choices=['baseline','route'],default='baseline')
    p.add_argument('--original',default=str(Path(__file__).resolve().parent/'data/original/data.csv'))
    p.add_argument('--leaves',type=int,default=63)
    p.add_argument('--teacher-csv',help='Original-only teacher probabilities, all train then test IDs')
    p.add_argument('--auxiliary-csv',help='Original-only auxiliary rating expectations, all train then test IDs')
    run(p.parse_args())
