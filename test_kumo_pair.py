"""Synthetic software checks only; no SDM, torch, model weights or real-data fits."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def runner():
    return importlib.import_module('train_kumo_pair')


def frame(rows=40):
    from train_tabpfn import FEATURES, CATEGORICAL
    data = {name: np.arange(rows, dtype=float) % 7 for name in FEATURES}
    for name in CATEGORICAL:
        data[name] = np.where(np.arange(rows) % 2, 'z', 'a')
    data['Flight Distance'] = np.arange(rows) % 4
    return pd.DataFrame(data)


def test_context_only_features_and_frozen_categories():
    m = runner(); x = frame(); y = np.arange(len(x)) % 2
    state, te = m.fit_preprocessor(x, y, 0)
    query = x.iloc[:3].copy()
    query.loc[query.index[0], 'Flight Distance'] = 999
    query.loc[query.index[0], 'Class'] = 'unseen'
    query.loc[query.index[1], 'Gender'] = None
    before = json.dumps(state, sort_keys=True, allow_nan=False)
    result = m.transform(query, state, 'fe')
    assert result.shape == (3, 29)
    assert result['Class'].isna().iloc[0]
    assert result['Gender'].isna().iloc[1]
    assert result['route_count'].iloc[0] == 0
    assert result['route_target'].iloc[0] == y.mean()
    assert result['route_mean_Class'].iloc[0] == .5
    assert state['categories']['Class'] == ['a', 'z']
    extra = pd.concat([query, frame().assign(**{'Class': 'other'})])
    pd.testing.assert_frame_equal(result.reset_index(drop=True), m.transform(extra, state, 'fe').iloc[:3].reset_index(drop=True))
    assert json.dumps(state, sort_keys=True, allow_nan=False) == before
    assert len(te) == len(x)


def test_crossfit_te_excludes_own_label_and_uses_inner_prior():
    m=runner(); x=frame(); y=np.arange(len(x))%2
    _, first=m.fit_preprocessor(x,y,2)
    for row in (0,7,19):
        changed=y.copy(); changed[row]=1-changed[row]
        _, second=m.fit_preprocessor(x,changed,2)
        assert first[row] == second[row]
    raw=m.transform(x,m.fit_preprocessor(x,y,2)[0],'raw')
    assert list(raw) == m.FEATURES


def test_selection_exact_order_shared_and_outer_disjoint():
    m=runner(); y=np.arange(200)%2; outer=np.arange(20,200)
    first=m.select_context(outer,y,100); second=m.select_context(outer,y,100)
    np.testing.assert_array_equal(first,second)
    assert len(first)==100 and len(set(first))==100 and set(first)<=set(outer)
    with pytest.raises(ValueError):m.select_context(outer,y,1000)


def test_input_hash_and_identity_guards(tmp_path):
    m=runner(); p=tmp_path/'x';p.write_text('safe')
    assert m.verify_hash(p,m.sha256(p)) == m.sha256(p)
    with pytest.raises(ValueError,match='SHA256'):m.verify_hash(p,'0'*64)
    x=frame(); train=x.assign(id=np.arange(len(x)),satisfaction=np.arange(len(x))%2)
    test=x.assign(id=np.arange(len(x))+len(x))
    m.validate_id_order(train,test)
    with pytest.raises(ValueError,match='IDs'):
        m.validate_id_order(train,test.assign(id=0))
    with pytest.raises(ValueError,match='order'):
        m.validate_id_order(train.iloc[::-1],test)


def test_fixed_dispatch_replay_and_failure():
    m=runner(); x=np.arange(1000*21,dtype=float).reshape(1000,21)
    class Model:
        fit_mode='fit_with_cache'
        def predict_proba(self,q):
            assert len(q)==128
            p=.1+.8*(q[:,0]/(1000*21))
            return np.column_stack([1-p,p])
    meta=m.calibration_metadata(); model=Model()
    pred=m.predict_fixed(model,x,meta['inference_dispatch'],nonconstant=True)
    m.verify_native(model,x,pred,meta)
    assert all(v['passed'] for v in meta['native_calibration']['comparisons'])
    class Drift(Model):
        def predict_proba(self,q):
            p=super().predict_proba(q);p[:,1]+=np.arange(len(q))*1e-5;p[:,0]=1-p[:,1];return p
    bad=Drift()
    with pytest.raises(AssertionError):m.verify_native(bad,x,m.predict_fixed(bad,x),m.calibration_metadata())


def test_complete_outputs_require_exact_coverage_and_order():
    m=runner(); n=30; testids=np.arange(100,111); ids=np.arange(n); y=ids%2; folds=ids%5
    oof=pd.DataFrame(dict(id=ids,satisfaction=y,fold=folds,raw=np.linspace(.1,.9,n),fe=np.linspace(.2,.8,n)))
    test=pd.DataFrame(dict(id=testids,raw=np.linspace(.1,.9,len(testids)),fe=np.linspace(.2,.8,len(testids))))
    m.validate_complete(oof,test,ids,y,folds,testids,[0,1,2,3,4])
    for broken in (oof.iloc[:-1],oof.iloc[::-1],oof.assign(fold=0),oof.assign(raw=np.nan)):
        with pytest.raises(ValueError):m.validate_complete(broken,test,ids,y,folds,testids,[0,1,2,3,4])
    with pytest.raises(ValueError):m.validate_complete(oof,test,ids,y,folds,testids,[0,1,2,3])


def test_packager_compiles_and_pins_native_artifacts(tmp_path):
    m=importlib.import_module('package_kumo_pair')
    out=m.package(tmp_path/'package')
    source=(out/'train.py').read_text();compile(source,'package','exec')
    metadata=json.loads((out/'kernel-metadata.json').read_text())
    assert metadata['id']=='wasidyms/s6e10-codex-kumo-paired-v1'
    assert metadata['is_private'] and metadata['enable_internet']
    assert metadata['competition_sources']==['playground-series-s6e10']
    assert 'fdc8372b510e19148f122681461438576916062a' in source
    assert '1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617' in source
    assert 'FIRST_PAIR_SECONDS = 3600' in source and 'TOTAL_BUDGET_SECONDS = 28800' in source


def test_supervisor_first_pair_timeout_preserves_evidence(tmp_path):
    m=importlib.import_module('package_kumo_pair')
    scope={'__name__':'test','RUNNER_SOURCE':'synthetic'};exec(m.SUPERVISOR,scope)
    import time
    result=scope['supervise'](tmp_path,"import time;from pathlib import Path;Path('partial').write_text('kept');time.sleep(20)",time.monotonic()+3,first_deadline=time.monotonic()+.2)
    assert result['status']=='timeout' and result['deadline_kind']=='first_pair'
    assert (tmp_path/'partial').read_text()=='kept'
    assert result['runtime_seconds']<2


def synthetic_run(monkeypatch, tmp_path, factory=None, rows=1000):
    import time
    from types import SimpleNamespace
    m=runner(); data=tmp_path/'data'; data.mkdir()
    train=frame(rows).assign(id=np.arange(rows), satisfaction=np.arange(rows)%2)
    test=frame(300).assign(id=np.arange(rows,rows+300))
    train.to_csv(data/'train.csv',index=False);test.to_csv(data/'test.csv',index=False)
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'synthetic')
    monkeypatch.setattr(m,'INPUT_HASHES',{name:m.sha256(data/name) for name in ('train.csv','test.csv')})
    monkeypatch.setattr(m,'CHECKPOINT_SHA',m.sha256(checkpoint))
    monkeypatch.setattr(m,'validate_inputs',m.validate_id_order)
    original_select=m.select_context
    monkeypatch.setattr(m,'select_context',lambda outer,y:original_select(outer,y,100))
    monkeypatch.setenv('KUMO_STARTED_MONOTONIC',str(time.monotonic()))
    class Model:
        fit_mode='fit_with_cache'
        def __init__(self,checkpoint,state,arm,fold): self.arm=arm
        def fit(self,x,y):pass
        def predict_proba(self,x):
            p=.1+.8*np.asarray(x[:,2],dtype=float)/6
            return np.column_stack([1-p,p])
        def clear(self):pass
        def archive(self):return {'synthetic':True}
    args=SimpleNamespace(data=str(data),checkpoint=str(checkpoint),output=str(tmp_path/'out'))
    return m,args,factory or Model


def test_full_synthetic_pair_coverage_and_marker(monkeypatch,tmp_path):
    m,args,factory=synthetic_run(monkeypatch,tmp_path)
    metadata=m.run(args, factory)
    out=Path(args.output)
    assert metadata['status']=='complete' and metadata['full_oof']
    assert metadata['completed_folds']==list(range(5))
    marker=json.loads((out/'first-fold-complete.json').read_text())
    assert all(arm['status']=='complete' for arm in marker['arms'].values())
    assert len(pd.read_csv(out/'oof.csv'))==1000
    assert len(pd.read_csv(out/'test.csv'))==300
    for fold in range(5):
        for arm in ('raw','fe'):
            assert (out/f'fold{fold}/{arm}_native_preprocessor.json').is_file()


def test_partial_arm_failure_has_no_first_pair_marker(monkeypatch,tmp_path):
    m,args,factory=synthetic_run(monkeypatch,tmp_path)
    class Fail(factory):
        def fit(self,x,y):
            if self.arm=='fe': raise RuntimeError('synthetic unsupported')
    with pytest.raises(RuntimeError,match='unsupported'):m.run(args,Fail)
    out=Path(args.output); metadata=json.loads((out/'metrics.json').read_text())
    assert metadata['status']=='failed' and not metadata['full_oof']
    assert metadata['folds'][0]['arms']['raw']['status']=='complete'
    assert metadata['folds'][0]['arms']['fe']['status']=='failed'
    assert (out/'fold0/raw_test.csv').is_file()
    assert not (out/'first-fold-complete.json').exists()
    assert not (out/'oof.csv').exists()


def test_native_transform_append_drift_rejects(monkeypatch):
    m=runner(); raw=frame(1000); state,_=m.fit_preprocessor(raw,np.arange(1000)%2,0)
    class Model:
        fit_mode='fit_with_cache'
        def predict_proba(self,x):
            p=.1+.8*np.asarray(x[:,2],dtype=float)/6
            return np.column_stack([1-p,p])
        def transformed(self,x):return [np.asarray(x[:,2:3],dtype=float)+len(x)*.001]
    model=Model(); x=m.transform(raw,state,'raw').to_numpy()
    with pytest.raises(AssertionError):m.verify_raw_replay(model,raw,state,'raw',m.predict_fixed(model,x),m.calibration_metadata())


def test_native_state_string_arrays_are_plain_json(monkeypatch):
    import types
    m=runner()
    class FakeTensor:pass
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(Tensor=FakeTensor))
    class Arrow:
        def to_pylist(self):return ['a','z']
    class StringTensor(FakeTensor):
        def to_arrow(self):return Arrow()
    encoded=m.native_state_json(StringTensor())
    assert encoded['arrow']==['a','z']
    json.dumps(encoded,allow_nan=False)


def test_total_deadline_after_first_pair_marker(tmp_path):
    import time
    m=importlib.import_module('package_kumo_pair'); scope={'__name__':'test','RUNNER_SOURCE':'synthetic'};exec(m.SUPERVISOR,scope)
    (tmp_path/'paired').mkdir();(tmp_path/'paired/first-fold-complete.json').write_text('{}')
    result=scope['supervise'](tmp_path,'import time;time.sleep(20)',time.monotonic()+.2,first_deadline=time.monotonic()-1)
    assert result['status']=='timeout' and result['deadline_kind']=='total'


def test_projection_failure_preserves_pair_without_marker(monkeypatch,tmp_path):
    import time
    m,args,factory=synthetic_run(monkeypatch,tmp_path)
    monkeypatch.setenv('KUMO_STARTED_MONOTONIC',str(time.monotonic()-4000))
    with pytest.raises(TimeoutError,match='cap'):m.run(args,factory)
    out=Path(args.output)
    assert not (out/'first-fold-complete.json').exists()
    metadata=json.loads((out/'metrics.json').read_text())
    assert metadata['status']=='failed' and metadata['completed_folds']==[0]
    assert all(a['status']=='complete' for a in metadata['folds'][0]['arms'].values())
    assert (out/'oof.partial.csv').is_file()


def test_native_schema_and_target_are_separate_context_fitted_tables():
    from contextlib import nullcontext
    from types import SimpleNamespace
    m=runner(); x=frame();state,te=m.fit_preprocessor(x,np.arange(len(x))%2,0)
    seen=[]
    def table(df,stypes,device):
        seen.append((df.copy(),stypes,device));return df
    model=m.NativeModel.__new__(m.NativeModel)
    model.state=state;model.columns=m.FEATURES+m.ENGINEERED
    model.sdm=SimpleNamespace(TableTensor=SimpleNamespace(from_pandas=table))
    model.torch=SimpleNamespace(amp=SimpleNamespace(autocast=lambda *args:nullcontext()),float16='fp16')
    model.generator='frozen';fits=[]
    model.model=SimpleNamespace(fit=lambda *args,**kwargs:fits.append((args,kwargs)))
    model.fit(m.transform(x,state,'fe',te).to_numpy(),np.arange(len(x))%2)
    query=x.iloc[:2].copy();query.loc[0,'Class']='unknown'
    model.table(m.transform(query,state,'fe').to_numpy())
    assert len(seen)==3
    context,target,queries=seen
    assert context[1]==queries[1]
    assert {name for name,stype in context[1].items() if stype=='categorical'}==set(m.CATEGORICAL)
    for name in m.CATEGORICAL:
        assert context[0][name].cat.categories.tolist()==queries[0][name].cat.categories.tolist()
    assert queries[0]['Class'].isna().iloc[0]
    assert target[0].satisfaction.cat.categories.tolist()==['0','1']
    assert fits[0][1]==dict(num_estimators=2,estimator_batch_size=1,generator='frozen')


def test_failed_input_receipt_is_preserved_before_model_construction(monkeypatch,tmp_path):
    m,args,factory=synthetic_run(monkeypatch,tmp_path)
    monkeypatch.setattr(m,'CHECKPOINT_SHA','0'*64)
    def forbidden(*args):raise AssertionError('model constructed')
    with pytest.raises(ValueError,match='SHA256'):m.run(args,forbidden)
    meta=json.loads((Path(args.output)/'metrics.json').read_text())
    assert meta['status']=='failed' and not meta['full_oof'] and not meta['folds']


def test_native_transform_missing_values_compare_equal():
    m=runner();raw=frame(1000);state,_=m.fit_preprocessor(raw,np.arange(1000)%2,0)
    class Model:
        fit_mode='fit_with_cache'
        def predict_proba(self,x):
            p=.1+.8*np.asarray(x[:,2],dtype=float)/6
            return np.column_stack([1-p,p])
        def transformed(self,x):
            result=np.asarray(x[:,2:3],dtype=float).copy();result[result==0]=np.nan
            return [result]
    model=Model();meta=m.calibration_metadata();x=m.transform(raw,state,'raw').to_numpy()
    m.verify_raw_replay(model,raw,state,'raw',m.predict_fixed(model,x),meta)
    json.dumps(meta,allow_nan=False)


def test_replay_failure_prevents_large_validation_and_test_calls(monkeypatch,tmp_path):
    m,args,factory=synthetic_run(monkeypatch,tmp_path,rows=6000)
    calls=[]; original=m.predict_fixed
    def predict(model,query,*args,**kwargs):
        calls.append(len(query));return original(model,query,*args,**kwargs)
    def fail(*args):raise AssertionError('synthetic native replay drift')
    monkeypatch.setattr(m,'predict_fixed',predict)
    monkeypatch.setattr(m,'verify_native',fail)
    with pytest.raises(AssertionError,match='replay drift'):m.run(args,factory)
    assert calls==[1000]
    out=Path(args.output)
    assert not (out/'fold0/raw_validation.csv').exists()
    assert not (out/'fold0/raw_test.csv').exists()
    assert not (out/'first-fold-complete.json').exists()
    assert json.loads((out/'metrics.json').read_text())['status']=='failed'
