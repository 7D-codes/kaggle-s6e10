"""Fold-safe features, using the competition-linked original dataset as a prior.
Ideas investigated in busyaprime/evgendvorkin/goodpjw2008 public S6E10 notebooks.
Implementation computes all competition population statistics on training folds only.
"""
import numpy as np
import pandas as pd


def encode_original_target(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        assert set(s.dropna().unique()).issubset({0,1,False,True})
        return s.astype(int)
    mapping={'satisfied':1,'neutral or dissatisfied':0,'true':1,'false':0}
    values=s.astype(str).str.lower().map(mapping)
    assert values.notna().all(), 'Unrecognized original target labels'
    return values.astype(int)


def add_features(a,b,t,original,kind):
    frames=[f.copy() for f in (a,b,t)]
    cols=list(a)
    oy=encode_original_target(original.satisfaction)
    orig=original[cols].copy()
    prior=float(oy.mean())
    for c in cols:
        stats=pd.DataFrame({'key':orig[c], 'target':oy}).groupby('key',dropna=False).target.agg(['sum','count'])
        rate=(stats['sum']+10*prior)/(stats['count']+10)
        for f in frames:
            f['orig_'+c]=f[c].map(rate).fillna(prior).astype('float32')
    if kind=='original':
        return frames
    assert kind=='rich'
    numeric=[c for c in cols if pd.api.types.is_numeric_dtype(a[c])]
    for c in cols:
        counts=a[c].value_counts(normalize=True,dropna=False)
        for f in frames:
            f['freq_'+c]=f[c].map(counts).fillna(0).astype('float32')
    for c in ('Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes'):
        if c not in a:
            continue
        for f in frames:
            values=f[c].fillna(-1)
            for scale in (1,10,100):
                f[f'digit_{c}_{scale}']=((values//scale)%10).astype('int8')
        if c=='Flight Distance':
            for f in frames:
                f['distance_bucket10']=(f[c]//10).astype('int32')
                f['distance_bucket100']=(f[c]//100).astype('int32')
                f['distance_category']=f[c].astype(str)
    # Route profiles use no labels and are fitted to the training fold only.
    key='Flight Distance'
    route=a.groupby(key)[[c for c in numeric if c!=key]].mean()
    for c in route:
        fallback=float(a[c].mean())
        for f in frames:
            mean=f[key].map(route[c]).fillna(fallback).astype('float32')
            f['route_mean_'+c]=mean
            f['route_diff_'+c]=(f[c]-mean).astype('float32')
    ratings=[c for c in numeric if c not in ('Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes')]
    for f in frames:
        r=f[ratings]
        f['rating_mean']=r.mean(axis=1).astype('float32')
        f['rating_std']=r.std(axis=1).astype('float32')
        f['rating_low_count']=(r<=2).sum(axis=1).astype('int8')
        f['rating_zero_count']=(r==0).sum(axis=1).astype('int8')
        for left,right in [('Online boarding','Inflight wifi service'),('Seat comfort','Inflight entertainment'),('Cleanliness','Food and drink')]:
            if left in f and right in f:
                f[f'equal_{left}_{right}']=(f[left]==f[right]).astype('int8')
                f[f'diff_{left}_{right}']=(f[left]-f[right]).astype('int8')
    return frames
