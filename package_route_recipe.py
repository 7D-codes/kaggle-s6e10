"""Package a bounded, fold-safe public-recipe benchmark. Never submits."""
from pathlib import Path
import json
import py_compile

root=Path(__file__).resolve().parent
out=root/'kaggle-route-recipe'
out.mkdir(exist_ok=True)
sources={name:(root/name).read_text() for name in
         ('features.py','external_teacher.py','recipe_features.py','train_baselines.py')}
script='''import subprocess,sys,json,gc
from pathlib import Path
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','lightgbm==4.7.0','xgboost==3.2.0','scikit-learn==1.9.1','pandas==3.1.0'],check=True)
root=Path('/kaggle/working');code=root/'reproducible-source';code.mkdir()
'''
script+='sources='+repr(sources)+'\n'
script+='''for name,source in sources.items():
    (code/name).write_text(source)
sys.path.insert(0,str(code))
data=next(Path('/kaggle/input').rglob('playground-series-s6e10/train.csv')).parent
original=next(Path('/kaggle/input').rglob('binary-aviation-satisfaction-129k/data.csv'))
teacher=root/'original-teacher'
subprocess.run([sys.executable,str(code/'external_teacher.py'),'--data',str(data),'--original',str(original),'--output',str(teacher),'--device','cuda','--rounds','600'],check=True)
print('Original-only CUDA teacher complete',flush=True)

# Real-data preprocessing round trip and tiny native-model acceptance checks.
import numpy as np,pandas as pd,joblib,lightgbm as lgb,xgboost as xgb
from recipe_features import prepare_recipe,transform_recipe
from external_teacher import exclude_competition_overlaps,require_cuda
from features import encode_original_target
from train_baselines import model_parameters
train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv')
source,removed=exclude_competition_overlaps(pd.read_csv(original),train,test)
assert removed==21
source['satisfaction']=encode_original_target(source.satisfaction)
raw_a=train.iloc[:256].drop(columns=['id','satisfaction'])
raw_b=train.iloc[256:296].drop(columns=['id','satisfaction'])
raw_t=test.iloc[:32].drop(columns='id')
a,b,t,state=prepare_recipe(raw_a,raw_b,raw_t,train.satisfaction.iloc[:256].astype(int),source,42,mode='full')
joblib.dump(state,root/'smoke-preprocessor.joblib',compress=3)
state=joblib.load(root/'smoke-preprocessor.joblib')
pd.testing.assert_frame_equal(b,transform_recipe(raw_b,state))
pd.testing.assert_frame_equal(t,transform_recipe(raw_t,state))
y=train.satisfaction.iloc[:256].astype(int);yb=train.satisfaction.iloc[256:296].astype(int)
lm=lgb.LGBMClassifier(**model_parameters('lightgbm','route',3,63,42,False));lm.fit(a,y)
assert np.isfinite(lm.predict_proba(b)).all()
xm=xgb.XGBClassifier(**model_parameters('xgboost','route',3,63,42,True))
xm.fit(a,y,eval_set=[(b,yb)],verbose=False);require_cuda(xm)
assert np.isfinite(xm.predict_proba(b)).all()
print('Full-view preprocessing round-trip, LightGBM and CUDA XGBoost smoke passed',flush=True)
del train,test,source,raw_a,raw_b,raw_t,a,b,t,state,lm,xm,y,yb;gc.collect()

def phase(name,features,model='lightgbm',rounds=8000,use_teacher=False):
    output=root/name
    command=[sys.executable,str(code/'train_baselines.py'),'--data',str(data),'--original',str(original),
        '--output',str(output),'--models',model,'--preset','route','--features',features,
        '--rounds',str(rounds),'--seed','42']
    if use_teacher:
        command+=['--teacher-csv',str(teacher/'teacher-features.csv')]
    if model=='xgboost':
        command+=['--gpu']
    subprocess.run(command,check=True)
    result=json.loads((output/'metrics.json').read_text())[model]
    print(json.dumps({'phase_complete':name,'oof_auc':result['oof_auc']}),flush=True)
    return result

control=phase('route-control','none')
te=phase('route-te-only','recipe_te_only')
full=phase('route-full','recipe_full',use_teacher=True)
gate=full['oof_auc']>=0.9602 and full['oof_auc']-control['oof_auc']>=0.0002
(root/'batch-summary.json').write_text(json.dumps({'control':control,'te_only':te,'full':full,
    'xgboost_gate_passed':gate,'submissions':0,'validation':'Development OOF, not an untouched holdout',
    'adaptation':'Seed42; train-fold-only population statistics and vocabularies; original-only XGB teacher; original overlaps excluded'},indent=2))
if gate:
    phase('route-full-xgboost','recipe_full',model='xgboost',rounds=12000,use_teacher=True)
else:
    print('GPU XGBoost extension skipped by predeclared performance gate',flush=True)
print('Bounded route-recipe benchmark completed; no submission made',flush=True)
'''
(out/'train.py').write_text(script)
py_compile.compile(str(out/'train.py'),doraise=True)
meta={
    'id':'wasidyms/s6e10-batch6-route-recipe',
    'title':'S6E10 Batch6 Route Recipe',
    'code_file':'train.py','language':'python','kernel_type':'script',
    'is_private':True,'enable_gpu':True,'enable_internet':True,
    'machine_shape':'NvidiaTeslaT4',
    'dataset_sources':['arseniyshutko/binary-aviation-satisfaction-129k'],
    'competition_sources':['playground-series-s6e10'],'kernel_sources':[],'model_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(out)
