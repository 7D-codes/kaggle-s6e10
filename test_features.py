import numpy as np
import pandas as pd
from features import encode_original_target, add_features


def test_boolean_target():
    assert encode_original_target(pd.Series([True,False])).tolist() == [1,0]
    assert encode_original_target(pd.Series(['satisfied','neutral or dissatisfied'])).tolist() == [1,0]


def test_train_only_statistics():
    base = pd.DataFrame({'Age':[20,20,30,40], 'Flight Distance':[100,100,200,300],
        'Gender':['M','M','F','F'], 'Online boarding':[1,2,3,4],
        'Inflight wifi service':[1,2,2,4], 'Seat comfort':[1,2,3,4],
        'Inflight entertainment':[1,3,3,4], 'Cleanliness':[1,2,3,4],
        'Food and drink':[1,2,2,4], 'Departure Delay in Minutes':[0,1,0,2],
        'Arrival Delay in Minutes':[0.,np.nan,0.,3.]})
    valid=base.iloc[[0]].copy()
    valid.loc[:,'Age']=99
    original=base.copy()
    original['satisfaction']=[True,False,True,False]
    a,b,t=add_features(base,valid,valid,original,'rich')
    assert a.shape[1]>base.shape[1]
    assert list(a)==list(b)==list(t)
    assert b['freq_Age'].iloc[0] == 0
    assert np.isclose(a['freq_Age'].iloc[0],0.5)
    assert np.isfinite(b['orig_Age'].iloc[0])
    # Validation values never enter the fitted vocabulary/statistics.
    valid.loc[:,'Age']=88
    a2,_,_=add_features(base,valid,valid,original,'rich')
    pd.testing.assert_frame_equal(a,a2)
