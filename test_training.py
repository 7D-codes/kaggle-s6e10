from types import SimpleNamespace
import pytest
from train_baselines import run


def test_refuse_to_overwrite_results(tmp_path):
    marker=tmp_path/'metrics.json'
    marker.write_text('{"preserve": true}')
    args=SimpleNamespace(output=str(tmp_path),data='/nonexistent',models='lightgbm')
    with pytest.raises(FileExistsError,match='fresh output'):
        run(args)
    assert marker.read_text() == '{"preserve": true}'


def test_route_preset_uses_reference_parameters_and_preserves_baseline():
    from train_baselines import model_parameters
    baseline=model_parameters('lightgbm','baseline',2000,63,42,False)
    route=model_parameters('lightgbm','route',8000,63,42,False)
    assert baseline['learning_rate']==0.04 and baseline['num_leaves']==63 and baseline['min_child_samples']==50
    assert route['learning_rate']==0.02 and route['num_leaves']==127 and route['max_depth']==8
    assert route['colsample_bytree']==0.5 and route['min_child_samples']==20 and route['n_estimators']==8000
    xgb=model_parameters('xgboost','route',12000,63,43,True)
    assert xgb['device']=='cuda' and xgb['max_depth']==8 and xgb['early_stopping_rounds']==300
    assert xgb['random_state']==43 and xgb['enable_categorical']


def test_auxiliary_csv_requires_aligned_bounded_rating_expectations(tmp_path):
    import pandas as pd
    from train_baselines import load_auxiliary_features
    path=tmp_path/'aux.csv'
    frame=pd.DataFrame({'id':[10,11,12],'aux_expected_Seat comfort':[1.2,2.3,4.5]})
    frame.to_csv(path,index=False)
    a,t=load_auxiliary_features(path,pd.Series([10,11]),pd.Series([12]))
    assert list(a)==['aux_expected_Seat comfort'] and a.iloc[1,0]==2.3 and t.iloc[0,0]==4.5
    for bad in (frame.iloc[::-1],frame.rename(columns={'aux_expected_Seat comfort':'satisfaction'}),
                frame.assign(**{'aux_expected_Seat comfort':[1.2,float('nan'),4.5]}),
                frame.assign(**{'aux_expected_Seat comfort':[1.2,6.1,4.5]})):
        bad.to_csv(path,index=False)
        with pytest.raises(ValueError):
            load_auxiliary_features(path,pd.Series([10,11]),pd.Series([12]))


def test_teacher_csv_requires_exact_ids_and_valid_probabilities(tmp_path):
    import numpy as np
    import pandas as pd
    from train_baselines import load_teacher_features
    path=tmp_path/'teacher.csv'
    frame=pd.DataFrame({'id':[10,11,12],'teacher_original_xgb':[0.2,0.4,0.7]})
    frame.to_csv(path,index=False)
    a,t=load_teacher_features(path,pd.Series([10,11]),pd.Series([12]))
    np.testing.assert_allclose(a,[0.2,0.4]);np.testing.assert_allclose(t,[0.7])
    for bad in (frame.iloc[::-1],frame.assign(teacher_original_xgb=[0.2,float('nan'),0.7]),
                frame.assign(teacher_original_xgb=[0.2,1.1,0.7])):
        bad.to_csv(path,index=False)
        with pytest.raises(ValueError):
            load_teacher_features(path,pd.Series([10,11]),pd.Series([12]))
