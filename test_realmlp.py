import pandas as pd
from train_realmlp import prepare

RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness']

def rating_fixture():
    a=pd.DataFrame({c:[0]*10 for c in RATINGS})
    a['Age']=[20,30]*5
    a['Flight Distance']=[100]*10
    a['Departure Delay in Minutes']=[0]*10
    a['Arrival Delay in Minutes']=[0]*10
    a['Gender']=['F']*10
    a['Customer Type']=['Loyal']*10
    a['Type of Travel']=['Business']*10
    a['Class']=['Eco']*10
    return a

def test_applicability_appends_only_raw_rating_count_and_mean():
    import numpy as np
    a=rating_fixture()
    b=a.iloc[[7,2,9]].copy()
    b.loc[7,RATINGS[0]]=1
    b.loc[7,RATINGS[1]]=0
    b.loc[7,RATINGS[2]]=3
    b.loc[2,RATINGS]=np.nan
    b.loc[9,RATINGS[2]]=5
    baseline=prepare(a,b,b,pd.Series([0,1]*5),42)
    enhanced=prepare(a,b,b,pd.Series([0,1]*5),42,features='applicability')
    for original,actual in zip(baseline,enhanced):
        pd.testing.assert_frame_equal(actual.iloc[:,:-2],original)
        assert list(actual.columns[-2:])==['rating_applicable_count','rating_applicable_mean']
    assert enhanced[1].index.tolist()==[7,2,9]
    assert enhanced[1]['rating_applicable_count'].tolist()==[2,0,1]
    assert enhanced[1]['rating_applicable_mean'].tolist()==[2.,0.,5.]
    assert enhanced[0]['rating_applicable_count'].eq(0).all()

def test_epoch_controls_preserve_fixed_refit_parameters():
    from train_realmlp import parse_args,realmlp_params
    default=parse_args(['--data','/unused'])
    assert (default.epochs,default.features)==(3,'baseline')
    original=realmlp_params(default.seed,default.epochs)
    assert (original['n_epochs'],original['stop_epoch'],original['n_ens'])==(3,3,8)
    changed=realmlp_params(default.seed,7)
    assert (changed['n_epochs'],changed['stop_epoch'])==(7,7)
    assert {k:v for k,v in changed.items() if k not in ('n_epochs','stop_epoch')}=={
        k:v for k,v in original.items() if k not in ('n_epochs','stop_epoch')}
    custom=parse_args(['--data','/unused','--epochs','7','--features','applicability'])
    assert (custom.epochs,custom.features)==(7,'applicability')

def test_epoch_controls_reject_nonpositive_or_nonnumeric_values():
    import pytest
    from train_realmlp import parse_args,realmlp_params
    for value in ('0','-1','abc','1.5'):
        with pytest.raises(SystemExit):
            parse_args(['--data','/unused','--epochs',value])
    for value in (0,-1):
        with pytest.raises(ValueError):
            realmlp_params(42,value)
    with pytest.raises(SystemExit):
        parse_args(['--data','/unused','--features','unexpected'])
    with pytest.raises(ValueError,match='feature mode'):
        prepare(rating_fixture(),None,None,None,42,features='unexpected')

def test_applicability_replays_exact_fold_features_without_heldout_leakage():
    from sklearn.model_selection import StratifiedKFold
    from predict_realmlp import reconstruct_features
    a=pd.concat([rating_fixture()]*5,ignore_index=True)
    a.loc[7,RATINGS[0]]=1
    a.loc[7,RATINGS[2]]=4
    a.loc[2,RATINGS]=float('nan')
    y=pd.Series([0,1]*25)
    train=a.assign(id=range(50),satisfaction=y)
    rows=a.iloc[[7,2,19]].copy()
    rows.index=[710,220,190]
    rows.loc[710,RATINGS[0]]=1
    ia,_=list(StratifiedKFold(5,shuffle=True,random_state=42).split(a,y))[2]
    _,expected,_=prepare(a.iloc[ia],rows,rows,y.iloc[ia],44,features='applicability')
    actual=reconstruct_features(train,rows,fold=2,seed=42,features='applicability')
    pd.testing.assert_frame_equal(actual,expected)
    assert actual.index.tolist()==[710,220,190]
    altered=rows.copy()
    altered.loc[710,RATINGS[0]]=5
    changed=reconstruct_features(train,altered,fold=2,seed=42,features='applicability')
    pd.testing.assert_frame_equal(actual.drop(index=710),changed.drop(index=710))
    baseline=reconstruct_features(train,rows,fold=2,seed=42)
    assert list(actual.columns[:-2])==list(baseline)
    with __import__('pytest').raises(ValueError,match='feature mode'):
        reconstruct_features(train,rows,fold=2,features='unsupported')

def test_checkpoint_feature_provenance_defaults_and_rejects_unknown(tmp_path):
    import json,pytest
    from predict_realmlp import validate_teacher_provenance
    metrics=tmp_path/'metrics.json'
    metrics.write_text(json.dumps({'provenance':{}}))
    assert validate_teacher_provenance(metrics,None)['feature_mode']=='baseline'
    metrics.write_text(json.dumps({'provenance':{'feature_mode':'applicability'}}))
    assert validate_teacher_provenance(metrics,None)['feature_mode']=='applicability'
    metrics.write_text(json.dumps({'provenance':{'feature_mode':'unknown'}}))
    with pytest.raises(ValueError,match='feature mode'):
        validate_teacher_provenance(metrics,None)

def test_cli_rejects_bad_feature_mode_before_reading_checkpoint(tmp_path):
    import json,subprocess,sys
    metrics=tmp_path/'metrics.json'
    metrics.write_text(json.dumps({'provenance':{'feature_mode':'unknown'}}))
    result=subprocess.run([sys.executable,'predict_realmlp.py','--checkpoint',str(tmp_path/'absent.pkl.gz'),
        '--train',str(tmp_path/'absent.csv'),'--rows',str(tmp_path/'absent-rows.csv'),
        '--fold','0','--metrics',str(metrics),'--output',str(tmp_path/'output.csv')],
        capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Unsupported feature mode' in result.stderr
    assert 'FileNotFoundError' not in result.stderr


def test_mlp_preprocessing_is_fold_local():
    a=pd.DataFrame({'Age':[20,20,30,40,20,30,40,50,60,70],
        'Flight Distance':[100]*10,'Class':['Eco']*10,'Type of Travel':['Business']*10,
        'Gender':['M']*10,'Food and drink':[3]*10,'Cleanliness':[3]*10,
        'Departure/Arrival time convenient':[2]*10,'Leg room service':[3]*10})
    b=a.iloc[:2].copy();b['Age']=99;b['Class']='Unseen'
    y=pd.Series([0,1]*5)
    xa,xb,xt=prepare(a,b,b,y,42)
    assert list(xa)==list(xb)==list(xt)
    assert xb['Age_cat_'].isna().all()
    assert (xb['_Class_count']==0).all()
    b['Age']=88
    xa2,_,_=prepare(a,b,b,y,42)
    pd.testing.assert_frame_equal(xa,xa2)


def test_checkpoint_preserves_local_schedule(tmp_path):
    import gzip
    import cloudpickle
    from train_realmlp import save_model
    factor=0.25
    model={'schedule':lambda epoch:factor*epoch}
    path=tmp_path/'model.pkl.gz'
    save_model(model,path)
    with gzip.open(path,'rb') as f:
        restored=cloudpickle.load(f)
    assert restored['schedule'](4)==1.0


def test_teacher_artifact_identity_is_checked(tmp_path):
    import hashlib,json
    import pytest
    from predict_realmlp import validate_teacher_provenance
    teacher=tmp_path/'teacher.csv'
    teacher.write_text('id,teacher_original_xgb\n1,0.2\n')
    metrics=tmp_path/'metrics.json'
    metrics.write_text(json.dumps({'provenance':{'teacher_csv_sha256':hashlib.sha256(teacher.read_bytes()).hexdigest()}}))
    validate_teacher_provenance(metrics,teacher)
    with pytest.raises(ValueError,match='required'):
        validate_teacher_provenance(metrics,None)
    teacher.write_text('id,teacher_original_xgb\n1,0.8\n')
    with pytest.raises(ValueError,match='SHA-256'):
        validate_teacher_provenance(metrics,teacher)
    metrics.write_text(json.dumps({'provenance':{}}))
    validate_teacher_provenance(metrics,None)
    with pytest.raises(ValueError,match='not trained'):
        validate_teacher_provenance(metrics,teacher)


def test_teacher_logit_feature_checks_alignment_and_keeps_frame_unchanged():
    import numpy as np
    import pytest
    from train_realmlp import add_teacher_logit
    frame=pd.DataFrame({'Age':[20,30]})
    enriched=add_teacher_logit(frame,np.array([0.2,0.8]))
    np.testing.assert_allclose(enriched.source_teacher_logit,np.log([0.25,4.]))
    assert list(frame)==['Age']
    with pytest.raises(ValueError):
        add_teacher_logit(frame,np.array([0.3]))
    with pytest.raises(ValueError):
        add_teacher_logit(frame,np.array([0.2,np.nan]))


def test_explicit_data_and_seed_work_outside_kaggle(tmp_path):
    from train_realmlp import parse_args
    args=parse_args(['--data',str(tmp_path),'--seed','2026'])
    assert args.data==str(tmp_path) and args.seed==2026


def test_saved_model_preprocessing_can_be_reconstructed():
    from sklearn.model_selection import StratifiedKFold
    from predict_realmlp import reconstruct_features
    a=pd.DataFrame({'Age':[20,30]*25,'Flight Distance':[100]*50,
        'Class':['Eco']*50,'Type of Travel':['Business']*50,'Gender':['M']*50,
        'Food and drink':[3]*50,'Cleanliness':[3]*50,
        'Departure/Arrival time convenient':[2]*50,'Leg room service':[3]*50})
    y=pd.Series([0,1]*25)
    train=a.assign(id=range(50),satisfaction=y)
    rows=a.iloc[:3].copy();rows['Age']=99
    split=list(StratifiedKFold(5,shuffle=True,random_state=42).split(a,y))
    ia,_=split[2]
    _,expected,_=prepare(a.iloc[ia],rows,rows,y.iloc[ia],44)
    actual=reconstruct_features(train,rows,fold=2,seed=42)
    pd.testing.assert_frame_equal(actual,expected)
    from train_realmlp import add_teacher_logit
    enriched=reconstruct_features(train,rows,fold=2,seed=42,teacher_prob=[0.2,0.5,0.8])
    pd.testing.assert_frame_equal(enriched,add_teacher_logit(expected,[0.2,0.5,0.8]))
