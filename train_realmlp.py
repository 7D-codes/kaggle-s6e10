"""Fold-local RealMLP adaptation of busyaprime/yekenot public S6E10 recipes.
Fixed-epoch training (three by default), no outer validation labels passed to the model.
No competition submission is made.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

COMBOS=[('Class','Type of Travel','Gender'),('Class','Type of Travel','Age'),
        ('Food and drink','Cleanliness'),('Flight Distance','Departure/Arrival time convenient')]
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking',
    'Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment',
    'On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']


def prepare(a,b,t,y,seed,features='baseline'):
    if features not in ('baseline','applicability'):
        raise ValueError(f'Unsupported feature mode: {features}')
    frames=[f.copy() for f in (a,b,t)]
    applicable=[]
    if features=='applicability':
        for raw in (a,b,t):
            rated=raw[RATINGS].where(raw[RATINGS].ne(0))
            applicable.append((rated.count(axis=1),rated.mean(axis=1).fillna(0)))
    cats=a.select_dtypes(include=['object','string','category']).columns.tolist()
    nums=[c for c in a if c not in cats]
    for f in frames:
        for c in cats:
            f[c]=f[c].astype('string').fillna('missing').astype(object)
        f[nums]=f[nums].fillna(0.)
    for c in nums:
        levels=pd.Index(frames[0][c].unique())
        for f in frames:
            f[c+'_cat_']=pd.Categorical(f[c],categories=levels)
    for c in cats+['Leg room service']:
        counts=frames[0][c].value_counts()
        for f in frames:
            f['_'+c+'_count']=f[c].map(counts).fillna(0).astype('int32')
    names=[]
    for combo in COMBOS:
        name='_'.join(combo)+'_'
        names.append(name)
        values=[f[list(combo)].astype(str).agg('_'.join,axis=1) for f in frames]
        levels=pd.Index(values[0].unique())
        for f,v in zip(frames,values):
            f[name]=pd.Categorical(v,categories=levels)
    enc=TargetEncoder(cv=5,smooth='auto',shuffle=True,random_state=seed)
    te_names=['_'+c+'TE' for c in names]
    frames[0][te_names]=enc.fit_transform(frames[0][names],y)
    for f in frames[1:]:
        f[te_names]=enc.transform(f[names])
    drop=['Class_Type of Travel_Age_','Flight Distance_Departure/Arrival time convenient_']
    prepared=[f.drop(columns=drop) for f in frames]
    if features=='applicability':
        for f,(count,mean) in zip(prepared,applicable):
            f['rating_applicable_count']=count
            f['rating_applicable_mean']=mean
    return prepared


def add_teacher_logit(frame,prob):
    prob=np.asarray(prob,dtype=float)
    if prob.shape!=(len(frame),) or not np.isfinite(prob).all() or not ((prob>=0)&(prob<=1)).all():
        raise ValueError('Teacher probabilities must align with the frame and be finite in [0,1]')
    clipped=np.clip(prob,1e-6,1-1e-6)
    return frame.assign(source_teacher_logit=np.log(clipped/(1-clipped)))


def save_model(model,path):
    import gzip
    import cloudpickle
    with gzip.open(path,'wb',compresslevel=3) as f:
        cloudpickle.dump(model,f)


def realmlp_params(seed,epochs):
    if isinstance(epochs,bool) or not isinstance(epochs,int) or epochs<=0:
        raise ValueError('epochs must be a positive integer')
    return dict(device='cuda',random_state=seed,verbosity=0,val_metric_name='1-auc_ovr',
        n_ens=8,n_epochs=epochs,stop_epoch=epochs,batch_size=256,use_early_stopping=False,
        lr=0.053,wd=0.015,sq_mom=0.988,lr_sched='flat_anneal',wd_sched='cos_log_15',
        first_layer_lr_factor=0.25,embedding_size=5,max_one_hot_cat_size=18,
        hidden_sizes=[512,256,128],act='silu',p_drop=0.05,p_drop_sched='invsqrtp1e-3',
        plr_hidden_1=16,plr_hidden_2=8,plr_act_name='gelu',plr_lr_factor=0.1151,
        plr_sigma=2.33,ls_eps=0.01,ls_eps_sched='sqrt_cos',add_front_scale=False,
        bias_init_mode='neg-uniform-dynamic-2',
        tfms=['one_hot','median_center','robust_scale','smooth_clip','embedding','l2_normalize'])

def run(args):
    import torch
    from pytabkit import RealMLP_TD_Classifier
    from importlib.metadata import version
    import gzip
    import cloudpickle
    assert torch.cuda.is_available(), 'Real CUDA GPU required'
    print('GPU:',torch.cuda.get_device_name(0),'PyTabKit:',version('pytabkit'),flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() or any(out.glob('realmlp-fold*')):
        raise FileExistsError('Use a fresh output directory')
    tr=pd.read_csv(Path(args.data)/'train.csv');te=pd.read_csv(Path(args.data)/'test.csv')
    sample=pd.read_csv(Path(args.data)/'sample_submission.csv')
    assert tr.id.is_unique and te.id.is_unique and sample.id.equals(te.id)
    X=tr.drop(columns=['id','satisfaction']);T=te.drop(columns='id')
    assert list(X)==list(T)
    y=tr.satisfaction.astype(int)
    teacher_path=getattr(args,'teacher_csv',None)
    if teacher_path is not None:
        from train_baselines import load_teacher_features
        teacher_train,teacher_test=load_teacher_features(teacher_path,tr.id,te.id)
    seed=getattr(args,'seed',42)
    params=realmlp_params(seed,args.epochs)
    # Exercise the exact no-validation refit path before allocating the full ensemble.
    sa,sb,st=prepare(X.iloc[:2048],X.iloc[2048:2058],T.iloc[:10],y.iloc[:2048],seed,features=args.features)
    if teacher_path is not None:
        sa=add_teacher_logit(sa,teacher_train[:2048])
        sb=add_teacher_logit(sb,teacher_train[2048:2058])
        st=add_teacher_logit(st,teacher_test[:10])
    smoke_params={**params,'n_ens':1,'n_epochs':1,'stop_epoch':1,'hidden_sizes':[32,16]}
    smoke=RealMLP_TD_Classifier(**smoke_params)
    smoke.fit(sa,y.iloc[:2048],val_idxs=np.empty(0,dtype=np.int64))
    smoke_pred=smoke.predict_proba(sb)
    assert np.isfinite(smoke_pred).all()
    smoke_path=out/'smoke-checkpoint.pkl.gz'
    save_model(smoke,smoke_path)
    with gzip.open(smoke_path,'rb') as f:
        restored=cloudpickle.load(f)
    np.testing.assert_allclose(restored.predict_proba(sb),smoke_pred,rtol=1e-6,atol=1e-7)
    del restored
    smoke_path.unlink()
    print('CUDA RealMLP fixed-epoch smoke test passed',flush=True)
    del smoke,sa,sb,st;gc.collect();torch.cuda.empty_cache()
    oof=np.zeros(len(y));pred=np.zeros(len(T));fold_ids=np.full(len(y),-1)
    scores=[];started=time.time()
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
        xa,xb,xt=prepare(X.iloc[a],X.iloc[b],T,y.iloc[a],seed+fold,features=args.features)
        if teacher_path is not None:
            xa=add_teacher_logit(xa,teacher_train[a])
            xb=add_teacher_logit(xb,teacher_train[b])
            xt=add_teacher_logit(xt,teacher_test)
        model=RealMLP_TD_Classifier(**{**params,'random_state':seed+fold})
        model.fit(xa,y.iloc[a],val_idxs=np.empty(0,dtype=np.int64))
        oof[b]=model.predict_proba(xb)[:,1];pred+=model.predict_proba(xt)[:,1]/5
        fold_ids[b]=fold;scores.append(float(roc_auc_score(y.iloc[b],oof[b])))
        save_model(model,out/f'realmlp-fold{fold}.pkl.gz')
        print(json.dumps({'fold':fold,'auc':scores[-1],'seconds':time.time()-started}),flush=True)
        del model,xa,xb,xt;gc.collect();torch.cuda.empty_cache()
    assert (fold_ids>=0).all() and np.isfinite(oof).all() and np.isfinite(pred).all()
    pd.DataFrame({'id':tr.id,'fold':fold_ids,'target':y,'prediction':oof}).to_csv(out/'realmlp-oof.csv',index=False)
    pd.DataFrame({'id':te.id,'satisfaction':pred}).to_csv(out/'realmlp-predictions.csv',index=False)
    result={'realmlp':{'oof_auc':float(roc_auc_score(y,oof)),'fold_auc':scores,
        'seconds':time.time()-started,'gpu':True,'parameters':params},
        'provenance':{'folds':5,'split_seed':seed,'feature_mode':args.features,'pytabkit_version':version('pytabkit'),
            'torch_version':torch.__version__,'pandas_version':pd.__version__,
            'cloudpickle_version':cloudpickle.__version__,'sklearn_version':version('scikit-learn'),
            'validation':'fixed epochs; outer validation labels never supplied to fit; development-exposed rows',
            'recipe_sources':['busyaprime/s6e10-route-or-distance-both-lb-0-96169','yekenot/ps-s6-e10-realmlp-pytabkit'],
            'submissions':0}}
    if teacher_path is not None:
        import hashlib
        result['provenance']['teacher_csv_sha256']=hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
        result['provenance']['teacher_feature']='original-only teacher, clipped logit, added after fold-local preparation'
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data')
    p.add_argument('--output',default='/kaggle/working')
    p.add_argument('--seed',type=int,default=42)
    def positive_int(value):
        try:
            number=int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError('epochs must be a positive integer') from exc
        if number<=0:
            raise argparse.ArgumentTypeError('epochs must be a positive integer')
        return number
    p.add_argument('--epochs',type=positive_int,default=3)
    p.add_argument('--features',choices=('baseline','applicability'),default='baseline')
    p.add_argument('--teacher-csv',help='Original-only teacher probabilities, all train then test IDs')
    args=p.parse_args(argv)
    if args.data is None:
        args.data=str(next(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv')).parent)
    return args


if __name__=='__main__':
    import subprocess,sys
    args=parse_args()
    subprocess.run([sys.executable,'-m','pip','install','-q','pytabkit==1.7.3','cloudpickle==3.1.2'],check=True)
    run(args)
