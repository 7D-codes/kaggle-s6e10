"""Synthetic software checks; never launch a competition fit."""
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest


def runner():
    return importlib.import_module('train_catboost_ctr')


def frame(rows=100):
    from recipe_features import CATS, NUMS, RATINGS
    x = pd.DataFrame({c: np.arange(rows, dtype=float) % 9 for c in NUMS + RATINGS})
    for c in CATS:
        x[c] = np.where(np.arange(rows) % 2, 'z', 'a')
    x['Flight Distance'] = np.arange(rows) % 4
    return x


def test_preprocessing_is_fit_only_json_and_exact_51_features():
    m=runner(); x=frame(); x.loc[0,'Arrival Delay in Minutes']=np.nan; x.loc[1,'Flight Distance']=np.nan
    state=m.fit_preprocessor(x)
    query=x.iloc[:3].copy();query.loc[0,'Flight Distance']=999;query.loc[0,'Class']='unseen';query.loc[1,'Class']=None
    before=json.dumps(state,sort_keys=True,allow_nan=False)
    result=m.transform(query,state,np.array([.1,.2,.3]))
    assert result.shape==(3,51) and len(m.CATEGORICAL)==21
    assert list(result)==m.FEATURE_ORDER and list(result)[:21]==m.NUMS+m.RATINGS+m.CATS
    assert result.loc[0,'route_count']==0 and result.loc[0,'count_Flight Distance']==0
    assert result.loc[0,'route_mean_Class']==.5
    assert result.loc[0,'Class']=='unseen' and result.loc[1,'Class']==m.MISSING
    assert np.isnan(result.loc[0,'Arrival Delay in Minutes'])
    assert result.loc[0,'cat_Arrival Delay in Minutes']=='-1'
    assert result.loc[0,'count_Arrival Delay in Minutes']==0
    assert result.loc[1,'route_count']==0
    assert result.loc[1,'route_mean_Class']==state['means']['Class']
    pd.testing.assert_frame_equal(result,m.transform(query,json.loads(before),[.1,.2,.3]))
    assert before==json.dumps(state,sort_keys=True,allow_nan=False)
    other=m.fit_preprocessor(x.iloc[:10]);assert state['counts']!=other['counts']


class FakeModel:
    calls=[]
    def __init__(self,**params):self.params=params; self.classes_=np.array([0,1]);self.__class__.calls.append(self)
    def fit(self,x,y,**kwargs):self.x=x.copy();self.y=np.array(y);self.fitargs=kwargs;return self
    def get_best_iteration(self):return 6
    def get_all_params(self):return self.params
    def save_model(self,path):Path(path).write_text('native')


def test_inner_selection_outer_refit_exact_and_no_outer_labels(tmp_path):
    m=runner();FakeModel.calls=[];x=frame();y=np.arange(100)%2;stages=[]
    model,state,receipt=m.fit_fold(x,y,np.linspace(.1,.9,100),2,tmp_path,lambda *v:stages.append(v),FakeModel)
    inner,outer=FakeModel.calls
    assert 'validation' not in inspect.signature(m.fit_fold).parameters
    assert len(inner.y)==80 and len(inner.fitargs['eval_set'][1])==20
    assert outer.params['iterations']==7 and outer.params['use_best_model'] is False
    assert outer.fitargs=={} and len(outer.y)==100
    assert inner.params['iterations']==20000 and inner.params['use_best_model'] is True
    assert inner.fitargs['early_stopping_rounds']==400
    for model in (inner,outer):
        assert model.params['random_seed']==11 and model.params['counter_calc_method']=='SkipTest'
    assert receipt['selected_iterations']==7
    assert json.loads((tmp_path/'inner-preprocessor.json').read_text())!=state
    assert [(name,cap) for name,cap in stages if cap is not None]==[('inner_fit',1800),('refit',1800)]
    assert (tmp_path/'inner.cbm').is_file() and (tmp_path/'outer.cbm').is_file()
    saved_receipt=json.loads((tmp_path/'selection.json').read_text())
    assert 'selection.json' not in saved_receipt['hashes']
    assert all(digest==m.sha256(tmp_path/name) for name,digest in saved_receipt['hashes'].items())


def test_class_order_and_native_replay_reject_drift():
    m=runner();x=frame(1000);state=m.fit_preprocessor(x);t=np.linspace(.1,.9,len(x))
    class Model:
        classes_=np.array(['1','0'])
        def predict_proba(self,q,**kwargs):
            assert kwargs==dict(task_type='CPU',thread_count=4)
            p=.1+.7*q['Age'].to_numpy()/9;return np.column_stack([p,1-p])
    model=Model();meta={'native_calibration':{'comparisons':[]}}
    anchor=m.verify_native(model,model,x,state,t,meta)
    assert np.ptp(anchor)>1e-6 and all(c['passed'] for c in meta['native_calibration']['comparisons'])
    model.classes_=[0,2]
    with pytest.raises(ValueError,match='classes'):m.predict(model,m.transform(x,state,t))


def test_teacher_hash_order_and_provenance_fail_before_constructor(tmp_path,monkeypatch):
    m=runner();root=tmp_path/'teacher';root.mkdir();producer=tmp_path/'external_teacher.py';producer.write_text('producer')
    for name in m.TEACHER_HASHES:(root/name).write_text('{}')
    with pytest.raises(ValueError,match='SHA256'):m.load_teacher(root,producer,np.arange(10),np.arange(10,14))
    monkeypatch.setattr(m,'verify_hash',lambda p,d:m.sha256(p))
    prov=dict(input_sha256=dict(original=m.ORIGINAL_SHA,train=m.INPUT_HASHES['train.csv'],test=m.INPUT_HASHES['test.csv']),original_exact_overlap_removed=21,original_rows_used=129859,configuration={'n_estimators':600},versions={'xgboost':'3.2.0'})
    (root/'provenance.json').write_text(json.dumps(prov))
    pd.DataFrame({'id':np.arange(14),'teacher_original_xgb':np.linspace(.1,.9,14)}).to_csv(root/'teacher-features.csv',index=False)
    a,b,_=m.load_teacher(root,producer,np.arange(10),np.arange(10,14));assert len(a)==10 and len(b)==4
    pd.DataFrame({'id':np.arange(14)[::-1],'teacher_original_xgb':np.linspace(.1,.9,14)}).to_csv(root/'teacher-features.csv',index=False)
    with pytest.raises(ValueError,match='IDs'):m.load_teacher(root,producer,np.arange(10),np.arange(10,14))
    prov['original_exact_overlap_removed']=20;(root/'provenance.json').write_text(json.dumps(prov))
    with pytest.raises(ValueError,match='provenance'):m.load_teacher(root,producer,np.arange(10),np.arange(10,14))


def test_complete_output_and_no_overwrite(tmp_path):
    m=runner();ids=np.arange(30);testids=np.arange(30,40);y=ids%2;fold=ids%5
    oof=pd.DataFrame(dict(id=ids,satisfaction=y,fold=fold,catboost=np.linspace(.1,.9,30)))
    test=pd.DataFrame(dict(id=testids,catboost=np.linspace(.1,.9,10)))
    m.validate_complete(oof,test,ids,y,fold,testids,list(range(5)))
    for broken in (oof.iloc[::-1],oof.iloc[:-1],oof.assign(catboost=np.nan),oof.assign(fold=0)):
        with pytest.raises(ValueError):m.validate_complete(broken,test,ids,y,fold,testids,list(range(5)))
    with pytest.raises(ValueError):m.validate_complete(oof,test,ids,y,fold,testids,[0])
    from argparse import Namespace
    with pytest.raises(FileExistsError):m.run(Namespace(output=str(tmp_path)))


def test_native_cpu_cbm_roundtrip_if_installed(tmp_path):
    cb=pytest.importorskip('catboost');m=runner();x=frame(1000);y=(x.Age.to_numpy()>4).astype(int);teacher=np.linspace(.1,.9,len(x))
    state=m.fit_preprocessor(x.iloc[:800]);values=m.transform(x.iloc[:800],state,teacher[:800])
    model=cb.CatBoostClassifier(iterations=12,depth=3,random_seed=11,verbose=False,allow_writing_files=False,thread_count=2,counter_calc_method='SkipTest')
    model.fit(values,y[:800],cat_features=m.CATEGORICAL);path=tmp_path/'native.cbm';model.save_model(path)
    reload=cb.CatBoostClassifier();reload.load_model(path)
    meta={'native_calibration':{'comparisons':[]}};m.verify_native(model,reload,x.iloc[800:],state,teacher[800:],meta)


def test_packager_frozen_metadata_and_compile(tmp_path):
    p=importlib.import_module('package_catboost_ctr');out=p.package(tmp_path/'package');meta=json.loads((out/'kernel-metadata.json').read_text())
    assert meta['id']=='wasidyms/s6e10-codex-catboost-ctr-v1' and meta['kernel_sources']==['wasidyms/s6e10-batch6-route-recipe']
    text=(out/'train.py').read_text();compile(text,'package','exec')
    for pin in ('catboost==1.2.8','pandas==2.3.3','numpy==2.1.3','scikit-learn==1.6.1'):assert pin in text
    assert 'phase_deadline_monotonic' in p.SUPERVISOR
    with pytest.raises(FileExistsError):p.package(out)


@pytest.mark.parametrize('kind',['phase','first_fold','total'])
def test_supervisor_kills_blocked_fit_and_descendants(tmp_path,kind):
    p=importlib.import_module('package_catboost_ctr');ns={'__name__':'test','RUNNER_SOURCE':'synthetic'};exec(p.SUPERVISOR,ns)
    worker="import json,os,time,subprocess,sys\nfrom pathlib import Path\nsubprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\nPath('stage.json').write_text(json.dumps(dict(stage='inner_fit',phase_deadline_monotonic=time.monotonic()+.12)))\ntime.sleep(60)" if kind=='phase' else "import time;time.sleep(60)"
    start=time.monotonic();res=ns['supervise'](tmp_path,worker,start+(.2 if kind=='total' else 3),first_deadline=start+(.2 if kind=='first_fold' else 3))
    assert res['status']=='timeout' and res['deadline_kind']==kind and time.monotonic()-start<2


def test_stage_cannot_clear_expired_fit_deadline(tmp_path,monkeypatch):
    m=runner();clock=[100.0];monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    path=tmp_path/'stage.json'
    m.stage_update(path,'inner_fit',1800,deadline=10000,first_deadline=5000)
    assert json.loads(path.read_text())['phase_deadline_monotonic']==1900
    clock[0]=1900.1
    with pytest.raises(TimeoutError,match='phase'):m.stage_update(path,'save',None,deadline=10000,first_deadline=5000)
    assert json.loads(path.read_text())['stage']=='inner_fit'


def test_failed_run_preserves_receipt_before_native_constructor(tmp_path,monkeypatch):
    m=runner();data=tmp_path/'data';data.mkdir();x=frame(100)
    x.assign(id=np.arange(100),satisfaction=np.arange(100)%2).to_csv(data/'train.csv',index=False)
    x.assign(id=np.arange(100,200)).to_csv(data/'test.csv',index=False)
    monkeypatch.setattr(m,'verify_hash',lambda p,h:m.sha256(p));monkeypatch.setattr(m,'validate_inputs',lambda *a:None)
    calls=[];monkeypatch.setattr(m,'native_factory',lambda **kw:calls.append(kw))
    from argparse import Namespace
    out=tmp_path/'output'
    with pytest.raises(FileNotFoundError):m.run(Namespace(data=data,teacher=tmp_path/'missing',producer=tmp_path/'missing.py',output=out))
    meta=json.loads((out/'metrics.json').read_text())
    assert meta['status']=='failed' and meta['full_oof'] is False and meta['completed_folds']==[] and calls==[]
    assert not (out/'oof.csv').exists() and not (out/'first-fold-complete.json').exists()


def test_native_replay_rejects_batch_and_serialization_drift():
    m=runner();x=frame(1000);state=m.fit_preprocessor(x);teacher=np.linspace(.1,.9,len(x))
    class Model:
        classes_=[0,1]
        def predict_proba(self,q,**kw):
            p=.1+.7*q.Age.to_numpy()/9
            return np.column_stack([1-p,p])
    class Drift(Model):
        def predict_proba(self,q,**kw):
            a=super().predict_proba(q,**kw);a[:,1]+=.00001*len(q);a[:,0]=1-a[:,1];return a
    with pytest.raises(AssertionError):m.verify_native(Model(),Drift(),x,state,teacher,{'native_calibration':{'comparisons':[]}})


def test_invalid_inner_iteration_never_refits(tmp_path):
    m=runner()
    class Bad(FakeModel):
        calls=[]
        def get_best_iteration(self):return -1
    with pytest.raises(ValueError,match='iteration'):
        m.fit_fold(frame(),np.arange(100)%2,np.linspace(.1,.9,100),0,tmp_path,lambda *a:None,Bad)
    assert len(Bad.calls)==1 and not (tmp_path/'outer.cbm').exists()


def test_generated_source_closure_imports_without_repository(tmp_path):
    import ast
    p=importlib.import_module('package_catboost_ctr');out=p.package(tmp_path/'package')
    namespace={'__name__':'test'};exec((out/'train.py').read_text(),namespace)
    source_assignment=ast.parse(namespace['WORKER_SOURCE']).body[0]
    sources=ast.literal_eval(source_assignment.value)
    isolated=tmp_path/'source-only';isolated.mkdir()
    for name,text in sources.items():(isolated/name).write_text(text)
    result=subprocess.run([sys.executable,'-c','import train_catboost_ctr as m; assert len(m.FEATURE_ORDER)==51; assert len(m.CATEGORICAL)==21'],cwd=isolated,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('fail_fold',[None,1])
def test_runner_full_coverage_or_partial_failure_receipt(tmp_path,monkeypatch,fail_fold):
    cb=pytest.importorskip('catboost');m=runner();data=tmp_path/'data';data.mkdir();x=frame(100)
    x.assign(id=np.arange(100),satisfaction=np.arange(100)%2).to_csv(data/'train.csv',index=False)
    x.assign(id=np.arange(100,200)).to_csv(data/'test.csv',index=False)
    teacher=tmp_path/'teacher';teacher.mkdir()
    for name in m.TEACHER_HASHES:(teacher/name).write_text('{}')
    producer=tmp_path/'producer.py';producer.write_text('synthetic')
    monkeypatch.setattr(m,'verify_hash',lambda p,h:m.sha256(p));monkeypatch.setattr(m,'validate_inputs',lambda *a:None)
    monkeypatch.setattr(m,'load_teacher',lambda *a:(np.linspace(.1,.9,100),np.linspace(.1,.9,100),{}))
    pins={'catboost':'1.2.8','numpy':'2.1.3','pandas':'2.3.3','scikit-learn':'1.6.1'}
    monkeypatch.setattr(m,'version',lambda name:pins[name])
    class Native(FakeModel):
        calls=[]
        def load_model(self,path):assert Path(path).exists();return self
        def predict_proba(self,q,**kw):
            p=.1+.7*q.Age.to_numpy()/9;return np.column_stack([1-p,p])
    monkeypatch.setattr(m,'native_factory',Native);monkeypatch.setattr(cb,'CatBoostClassifier',Native)
    real_fit=m.fit_fold
    def fit(*args,**kw):
        if args[3]==fail_fold:raise RuntimeError('synthetic interruption')
        return real_fit(*args,**kw)
    monkeypatch.setattr(m,'fit_fold',fit)
    from argparse import Namespace
    out=tmp_path/'out';args=Namespace(data=data,teacher=teacher,producer=producer,output=out)
    if fail_fold is not None:
        with pytest.raises(RuntimeError,match='interruption'):m.run(args)
        meta=json.loads((out/'metrics.json').read_text())
        assert meta['status']=='failed' and not meta['full_oof'] and meta['completed_folds']==[0]
        assert (out/'oof.partial.csv').exists() and not (out/'oof.csv').exists()
    else:
        meta=m.run(args);assert meta['full_oof'] and meta['completed_folds']==list(range(5))
        result=pd.read_csv(out/'oof.csv');assert len(result)==100 and np.array_equal(result.id,np.arange(100))
        assert result.fold.value_counts().to_dict()=={i:20 for i in range(5)}
        assert (out/'first-fold-complete.json').exists() and len(pd.read_csv(out/'test.csv'))==100
        for receipt in meta['folds']:
            assert receipt['hashes']['selection.json']==m.sha256(out/f"fold{receipt['fold']}"/'selection.json')


def test_native_factory_rejects_cpu_without_constructing(monkeypatch):
    cb=pytest.importorskip('catboost');m=runner();import catboost.utils
    monkeypatch.setattr(catboost.utils,'get_gpu_device_count',lambda:0)
    monkeypatch.setattr(m,'version',lambda name:'1.2.8')
    calls=[];monkeypatch.setattr(cb,'CatBoostClassifier',lambda **kw:calls.append(kw))
    with pytest.raises(RuntimeError,match='no CPU'):m.native_factory(**m.MODEL_PARAMETERS)
    assert calls==[]


def test_native_cpu_fallback_resolved_params_fail_closed(tmp_path,monkeypatch):
    m=runner()
    class CPU(FakeModel):
        def get_all_params(self):return {**self.params,'task_type':'CPU'}
    monkeypatch.setattr(m,'native_factory',CPU)
    with pytest.raises(RuntimeError,match='did not fit on GPU'):
        m.fit_fold(frame(),np.arange(100)%2,np.linspace(.1,.9,100),0,tmp_path,lambda *a:None)
    assert not (tmp_path/'outer.cbm').exists()
