"""Fixed native CatBoost CTR component. Evidence outputs only; never submits."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from recipe_features import CATS, NUMS, RATINGS
from train_baselines import load_teacher_features
from train_kumo_pair import INPUT_HASHES, validate_inputs, verify_hash
from train_tabpfn import compare_predictions, probabilities, sha256

RAW = NUMS + RATINGS + CATS
GROUP_MEANS = [c for c in RAW if c != 'Flight Distance']
CATEGORICAL = CATS + RATINGS + ['cat_' + c for c in NUMS]
FEATURE_ORDER = RAW + ['route_mean_' + c for c in GROUP_MEANS] + ['route_count'] + ['count_' + c for c in NUMS] + ['teacher_original_xgb'] + ['cat_' + c for c in NUMS]
MISSING = '__CATBOOST_MISSING__'
SOURCE_NOTEBOOK = 'goodpjw2008/s6e10-catboost-ctr-glm-margin-xgb-lb-0-96152'
SOURCE_TEXT_SHA = 'c49fd0c3dc91799c1619aa9bd9338528e4812510efd0ae48d8489ad6d9794d41'
MODEL_PARAMETERS = dict(depth=6, l2_leaf_reg=10, learning_rate=.05, random_seed=11,
    border_count=254, loss_function='Logloss', eval_metric='Logloss', task_type='GPU', devices='0',
    thread_count=4, verbose=False, allow_writing_files=False, counter_calc_method='SkipTest')
TEACHER_HASHES = {'teacher-features.csv':'d12e7965b26df3b243dcecb0a704d5effad2346f0c56bdaeb1105f90845fc290',
    'provenance.json':'5b95ba97b2519ec534742bd3764f68a1d669b66f4263603a3e1706b359f1f964',
    'teacher-model.json':'46cebda64203f9701c73020669df66ef9b42a9530eebd9ca748db5e508e4f28b'}
PRODUCER_SHA = 'dbb6107e434d4da7e2e54fd3e6ad7a743688ac359810c0dc4f8d64bed6731a89'
ORIGINAL_SHA = '2ad76465f6115f93d16121891a6b1235acb8007a4ab065cae0dc46b60ad116e3'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def stage_update(path, name, seconds, *, deadline, first_deadline=None):
    """Publish atomically and never clear an expired blocking-fit deadline."""
    now = time.monotonic()
    if now >= deadline or (first_deadline is not None and now >= first_deadline):
        raise TimeoutError('Frozen total/first-fold deadline exhausted')
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text()).get('phase_deadline_monotonic')
        if previous is not None and now >= previous:
            raise TimeoutError('Frozen fit phase deadline exhausted')
    write_json(path, dict(stage=name, phase_deadline_monotonic=None if seconds is None else min(deadline, now + seconds)))


def keys(values):
    return values.map(lambda v: '__missing__' if pd.isna(v) else str(float(v)))


def fit_preprocessor(raw):
    raw = raw[RAW].reset_index(drop=True)
    state = dict(categories={c: sorted(raw[c].dropna().unique().tolist()) for c in CATS}, means={}, routes={}, frequencies={})
    route = keys(raw['Flight Distance'])
    for c in GROUP_MEANS:
        values = raw[c]
        if c in CATS:
            values = values.map({v:i for i,v in enumerate(state['categories'][c])}).fillna(-1)
        values = values.astype(float)
        mean = values.mean()
        if pd.isna(mean):
            raise ValueError(f'No fitted nonmissing values for group mean {c}')
        state['means'][c] = float(mean)
        state['routes'][c] = values[raw['Flight Distance'].notna()].groupby(route).mean().dropna().to_dict()
    state['counts'] = {k:int(v) for k,v in route[raw['Flight Distance'].notna()].value_counts().items()}
    for c in NUMS:
        state['frequencies'][c] = {k:int(v) for k,v in keys(raw[c].dropna()).value_counts().items()}
    return state


def transform(raw, state, teacher):
    teacher = np.asarray(teacher, dtype=float)
    if teacher.shape != (len(raw),) or not np.isfinite(teacher).all() or ((teacher < 0) | (teacher > 1)).any():
        raise ValueError('Expected aligned finite bounded teacher input')
    result = raw[RAW].reset_index(drop=True).copy()
    route = keys(result['Flight Distance'])
    for c in GROUP_MEANS:
        result['route_mean_' + c] = route.map(state['routes'][c]).fillna(state['means'][c])
    result['route_count'] = route.map(state['counts']).fillna(0)
    for c in NUMS:
        result['count_' + c] = keys(result[c]).map(state['frequencies'][c]).fillna(0)
    result['teacher_original_xgb'] = teacher
    for c in NUMS:
        result['cat_' + c] = result[c].fillna(-1).astype('int64').astype(str)
    for c in CATS + RATINGS:
        result[c] = result[c].map(lambda v: MISSING if pd.isna(v) else str(v))
    return result[FEATURE_ORDER]


def load_teacher(directory, producer, train_ids, test_ids):
    directory = Path(directory)
    hashes = {name:verify_hash(directory / name,digest) for name,digest in TEACHER_HASHES.items()}
    hashes['producer'] = verify_hash(producer, PRODUCER_SHA)
    p = json.loads((directory / 'provenance.json').read_text())
    if (p.get('input_sha256') != dict(original=ORIGINAL_SHA, train=INPUT_HASHES['train.csv'], test=INPUT_HASHES['test.csv'])
            or p.get('original_exact_overlap_removed') != 21 or p.get('original_rows_used') != 129859
            or p.get('configuration', {}).get('n_estimators') != 600 or p.get('versions', {}).get('xgboost') != '3.2.0'):
        raise ValueError('Qualified teacher provenance mismatch')
    a,b = load_teacher_features(directory / 'teacher-features.csv',pd.Series(train_ids),pd.Series(test_ids))
    ids = np.concatenate([train_ids,test_ids])
    if len(np.unique(ids)) != len(ids) or np.ptp(np.concatenate([a,b])) <= 1e-6:
        raise ValueError('Teacher IDs/order or nonconstant probabilities invalid')
    return a,b,hashes


def native_factory(**parameters):
    # Immutable input qualification happens in run before any native import/constructor.
    from catboost import CatBoostClassifier
    from catboost.utils import get_gpu_device_count
    count = get_gpu_device_count()
    if version('catboost') != '1.2.8' or count < 1:
        raise RuntimeError('CatBoost1.2.8 and actual GPU required; no CPU training fallback')
    model = CatBoostClassifier(**parameters)
    model._codex_device = dict(gpu_count=count, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        details = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
            '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=10)
        model._codex_device['nvidia_smi'] = details.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        model._codex_device['nvidia_smi_error'] = str(error)
    return model


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def fit_fold(raw, labels, teacher, fold, directory, stage, model_factory=None):
    """Only outer-training rows/labels enter this helper; outer y is never accepted."""
    directory = Path(directory)
    labels = np.asarray(labels);teacher = np.asarray(teacher)
    inner,held = train_test_split(np.arange(len(raw)),test_size=.2,stratify=labels,random_state=20261026+fold)
    receipt = dict(inner_seed=20261026+fold, inner_indices_sha256=index_hash(inner), inner_holdout_indices_sha256=index_hash(held))
    factory = model_factory or native_factory
    state = fit_preprocessor(raw.iloc[inner])
    write_json(directory / 'inner-preprocessor.json',state)
    model = factory(**MODEL_PARAMETERS,iterations=20000,use_best_model=True,cat_features=CATEGORICAL)
    stage('inner_fit',1800)
    started=time.monotonic()
    model.fit(transform(raw.iloc[inner],state,teacher[inner]),labels[inner],
        eval_set=(transform(raw.iloc[held],state,teacher[held]),labels[held]),early_stopping_rounds=400)
    stage('inner_save',None)
    receipt['inner_fit_seconds']=time.monotonic()-started
    selected=int(model.get_best_iteration())+1
    if not 1 <= selected <= 20000:raise ValueError('Invalid selected inner iteration')
    receipt.update(selected_iterations=selected,inner_parameters=model.get_all_params(),
        inner_device=getattr(model, '_codex_device', None))
    if model_factory is None and receipt['inner_parameters'].get('task_type')!='GPU':raise RuntimeError('Inner model did not fit on GPU')
    model.save_model(directory / 'inner.cbm')
    if hasattr(model,'get_evals_result'):receipt['inner_evaluations']=model.get_evals_result()
    write_json(directory / 'selection.json',receipt)
    del model
    state=fit_preprocessor(raw)
    write_json(directory / 'outer-preprocessor.json',state)
    model=factory(**MODEL_PARAMETERS,iterations=selected,use_best_model=False,cat_features=CATEGORICAL)
    stage('refit',1800)
    started=time.monotonic()
    model.fit(transform(raw,state,teacher),labels)
    stage('outer_save',None)
    receipt['refit_fit_seconds']=time.monotonic()-started
    receipt['outer_parameters']=model.get_all_params()
    receipt['outer_device']=getattr(model, '_codex_device', None)
    if model_factory is None and receipt['outer_parameters'].get('task_type')!='GPU':raise RuntimeError('Outer model did not fit on GPU')
    model.save_model(directory / 'outer.cbm')
    receipt['hashes']={p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p.name != 'selection.json'}
    write_json(directory / 'selection.json',receipt)
    return model,state,receipt


def predict(model, values, nonconstant=False):
    classes = [str(c) for c in model.classes_]
    if len(classes)!=2 or set(classes)!={'0','1'}:raise ValueError('Expected native classes 0/1')
    raw=np.asarray(model.predict_proba(values,task_type='CPU',thread_count=4))[:,[classes.index('0'),classes.index('1')]]
    return probabilities(raw,len(values),nonconstant=nonconstant)


def verify_native(model, reloaded, raw, state, teacher, receipt):
    values=transform(raw,state,teacher)
    anchor=predict(model,values,nonconstant=True)
    compare_predictions(receipt,'native_cbm_reload',predict(reloaded,values,True),anchor)
    compare_predictions(receipt,'repeat',predict(model,values,True),anchor)
    rng=np.random.default_rng(42)
    for size in (1,17,63,65,127,128,129):
        positions=rng.permutation(len(values))[:size]
        compare_predictions(receipt,f'subset{size}',predict(reloaded,values.iloc[positions]),anchor[positions])
    positions=rng.permutation(len(values))
    compare_predictions(receipt,'permutation',predict(reloaded,values.iloc[positions])[np.argsort(positions)],anchor)
    independent=transform(raw.copy(),json.loads(json.dumps(state)),teacher)
    compare_predictions(receipt,'raw_preprocessor_json',predict(reloaded,independent,True),anchor)
    return anchor


def validate_complete(oof, testing, ids, labels, folds, test_ids, completed):
    if completed!=list(range(5)):raise ValueError('Incomplete folds')
    for actual,expected in ((oof.id,ids),(oof.satisfaction,labels),(oof.fold,folds),(testing.id,test_ids)):
        if not np.array_equal(actual,expected):raise ValueError('Incomplete or mismatched output order/coverage')
    for frame in (oof,testing):
        p=frame.catboost.to_numpy(dtype=float)
        if frame.id.duplicated().any() or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or np.ptp(p)<=1e-6:
            raise ValueError('Invalid or constant complete output')


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=float(os.environ.get('CATBOOST_STARTED_MONOTONIC',time.monotonic()))
    deadline=float(os.environ.get('PILOT_DEADLINE_MONOTONIC',started+21600))
    stage_path=Path(os.environ.get('CATBOOST_STAGE_PATH',output/'stage.json'))
    metadata=dict(status='running',full_oof=False,submissions=0,completed_folds=[],folds=[],model_parameters=MODEL_PARAMETERS,
        feature_order=FEATURE_ORDER,categorical=CATEGORICAL,versions={'python':platform.python_version()},
        teacher_inference='frozen qualified upstream input; historical teacher pipeline not regenerated',
        source_notebook=SOURCE_NOTEBOOK,source_text_sha256=SOURCE_TEXT_SHA)
    def save():
        metadata['runtime_seconds']=time.monotonic()-started;write_json(output/'metrics.json',metadata)
    def stage(name,seconds=None):
        stage_update(stage_path, name, seconds, deadline=deadline,
            first_deadline=None if metadata['completed_folds'] else started + 3600)
    save()
    try:
        stage('input_validation')
        data=Path(args.data)
        metadata['hashes']={name:verify_hash(data/name,digest) for name,digest in INPUT_HASHES.items()}
        metadata['hashes']['source']=sha256(__file__)
        train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');validate_inputs(train,test)
        ta,tt,hashes=load_teacher(args.teacher,args.producer,train.id.to_numpy(),test.id.to_numpy())
        metadata['teacher_hashes']=hashes
        upstream=output/'original-teacher';upstream.mkdir()
        for name in TEACHER_HASHES:shutil.copyfile(Path(args.teacher)/name,upstream/name)
        shutil.copyfile(args.producer,upstream/'external_teacher.py')
        for name in ('catboost','numpy','pandas','scikit-learn'):metadata['versions'][name]=version(name)
        expected_versions={'catboost':'1.2.8','numpy':'2.1.3','pandas':'2.3.3','scikit-learn':'1.6.1'}
        if any(metadata['versions'][k]!=v for k,v in expected_versions.items()):raise RuntimeError('Frozen runtime version mismatch')
        metadata['configuration_sha256']=hashlib.sha256(json.dumps(dict(parameters=MODEL_PARAMETERS,features=FEATURE_ORDER,categorical=CATEGORICAL,split_seed=42),sort_keys=True).encode()).hexdigest()
        labels=train.satisfaction.to_numpy(dtype=int)
        splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,labels));folds=np.full(len(train),-1)
        for fold,(_,valid) in enumerate(splits):folds[valid]=fold
        oof=pd.DataFrame(dict(id=train.id,satisfaction=labels,fold=np.full(len(train),-1),catboost=np.nan));test_sum=np.zeros(len(test))
        for fold,(fitting,valid) in enumerate(splits):
            directory=output/f'fold{fold}';directory.mkdir()
            receipt=dict(fold=fold,status='running',outer_train_indices_sha256=index_hash(fitting),validation_indices_sha256=index_hash(valid),native_calibration={'comparisons':[]})
            metadata['folds'].append(receipt);save()
            model,state,selection=fit_fold(train.iloc[fitting],labels[fitting],ta[fitting],fold,directory,stage)
            receipt.update(selection);save();stage('native_calibration')
            from catboost import CatBoostClassifier
            reloaded=CatBoostClassifier();reloaded.load_model(directory/'outer.cbm')
            anchor=verify_native(model,reloaded,train.iloc[valid[:1000]],state,ta[valid[:1000]],receipt)
            stage('validation_prediction')
            prediction=predict(reloaded,transform(train.iloc[valid],state,ta[valid]),True)
            compare_predictions(receipt,'full_validation_prefix',prediction[:len(anchor)],anchor)
            stage('test_prediction')
            testing=predict(reloaded,transform(test,state,tt),True)
            pd.DataFrame(dict(id=train.id.iloc[valid],catboost=prediction)).to_csv(directory/'validation.csv',index=False)
            pd.DataFrame(dict(id=test.id,catboost=testing)).to_csv(directory/'test.csv',index=False)
            # Outer validation labels are first used here, after all predictions and native gates.
            receipt.update(status='complete',auc=float(roc_auc_score(labels[valid],prediction)),prediction_device='CPU',training_device=receipt['outer_parameters']['task_type'])
            receipt['hashes'].update({p.name:sha256(p) for p in directory.iterdir() if p.is_file()})
            oof.loc[valid,'catboost']=prediction;oof.loc[valid,'fold']=fold;test_sum+=testing
            oof.to_csv(output/'oof.partial.csv',index=False)
            elapsed=time.monotonic()-started
            if fold==0:
                metadata.update(first_fold_seconds=elapsed,projected_total_seconds=1.25*5*elapsed)
                if elapsed>3600 or 1.25*5*elapsed>21600:raise TimeoutError('First-fold time/projection exceeds frozen cap')
                write_json(output/'first-fold-complete.json',dict(complete=True,elapsed_seconds=elapsed))
            metadata['completed_folds'].append(fold);save()
            del model,reloaded
        stage('complete_validation')
        testing=pd.DataFrame(dict(id=test.id,catboost=test_sum/5))
        validate_complete(oof,testing,train.id.to_numpy(),labels,folds,test.id.to_numpy(),metadata['completed_folds'])
        oof.to_csv(output/'oof.csv',index=False);testing.to_csv(output/'test.csv',index=False)
        metadata.update(status='complete',full_oof=True,auc=float(roc_auc_score(labels,oof.catboost)),output_hashes={name:sha256(output/name) for name in ('oof.csv','test.csv')})
        stage('complete');save();return metadata
    except BaseException as error:
        metadata.update(status='failed',full_oof=False,error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','teacher','producer','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
