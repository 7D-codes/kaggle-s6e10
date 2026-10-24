"""Reload a trusted RealMLP fold checkpoint and reconstruct its raw-row preprocessing.

The checkpoint is intentionally paired with the original train.csv and train_realmlp.py.
Use the training job's Python/pandas/sklearn/pytabkit/torch versions and a CUDA GPU.
Never load a checkpoint from an untrusted source, cloudpickle executes code.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from train_realmlp import prepare,add_teacher_logit


def validate_teacher_provenance(metrics_path,teacher_path):
    provenance=json.loads(Path(metrics_path).read_text())['provenance']
    mode=provenance.get('feature_mode','baseline')
    if mode not in ('baseline','applicability'):
        raise ValueError(f'Unsupported feature mode: {mode}')
    provenance['feature_mode']=mode
    expected=provenance.get('teacher_csv_sha256')
    if expected is not None:
        if teacher_path is None:
            raise ValueError('Teacher CSV required for this checkpoint')
        if hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Teacher CSV SHA-256 differs from training')
    elif teacher_path is not None:
        raise ValueError('Checkpoint was not trained with teacher features')
    return provenance


def reconstruct_features(train,rows,fold,seed=42,teacher_prob=None,features='baseline'):
    if features not in ('baseline','applicability'):
        raise ValueError(f'Unsupported feature mode: {features}')
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    assert train.id.is_unique and train.satisfaction.notna().all()
    X=train.drop(columns=['id','satisfaction'])
    y=train.satisfaction.astype(int)
    raw=rows.drop(columns=['id'],errors='ignore')
    assert list(raw)==list(X), 'Raw feature names and order must match training'
    ia,_=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y))[fold]
    _,transformed,_=prepare(X.iloc[ia],raw,raw,y.iloc[ia],seed+fold,features=features)
    return transformed if teacher_prob is None else add_teacher_logit(transformed,teacher_prob)


if __name__=='__main__':
    import cloudpickle
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--train',required=True)
    p.add_argument('--rows',required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--teacher-csv',help='Required for teacher-enhanced checkpoints; same teacher used in fitting')
    p.add_argument('--metrics',help='Training metrics.json; defaults to the checkpoint directory')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():
        raise FileExistsError('Use a fresh prediction output path')
    provenance=validate_teacher_provenance(a.metrics or Path(a.checkpoint).parent/'metrics.json',a.teacher_csv)
    if a.seed!=provenance.get('split_seed',42):
        raise ValueError('Prediction seed differs from checkpoint training split')
    train=pd.read_csv(a.train);rows=pd.read_csv(a.rows)
    teacher=None
    if a.teacher_csv is not None:
        features=pd.read_csv(a.teacher_csv)
        assert features.id.is_unique
        teacher=features.set_index('id').loc[rows.id,'teacher_original_xgb'].to_numpy()
    transformed=reconstruct_features(train,rows,a.fold,a.seed,teacher,features=provenance['feature_mode'])
    with gzip.open(a.checkpoint,'rb') as f:
        model=cloudpickle.load(f)
    prob=model.predict_proba(transformed)[:,1]
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    pd.DataFrame({'id':rows.id,'satisfaction':prob}).to_csv(out,index=False)
    print(f'Wrote {len(prob)} single-fold predictions to {out}')
