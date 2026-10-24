from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT / 'data/train.csv')
test = pd.read_csv(ROOT / 'data/test.csv')
submission = pd.read_csv(ROOT / 'data/sample_submission.csv')
features = [c for c in train if c not in ('id', 'satisfaction')]
assert set(features) == set(test.columns) - {'id'}
assert train.id.is_unique and test.id.is_unique
assert set(train.id).isdisjoint(test.id)
assert submission.id.equals(test.id)
assert set(submission.columns) == {'id', 'satisfaction'}
report = {
    'train_rows': len(train), 'test_rows': len(test),
    'feature_count': len(features),
    'target_counts': train.satisfaction.value_counts(dropna=False).to_dict(),
    'target_proportions': train.satisfaction.value_counts(normalize=True).to_dict(),
    'target_missing': int(train.satisfaction.isna().sum()),
    'features': {c: {'dtype': str(train[c].dtype), 'unique_train': int(train[c].nunique()),
                     'missing_train': int(train[c].isna().sum()),
                     'missing_test': int(test[c].isna().sum())} for c in features},
    'categorical_values': {c: train[c].value_counts().to_dict() for c in features if pd.api.types.is_string_dtype(train[c])},
    'duplicate_feature_rows': int(train.duplicated(subset=features).sum()),
    'train_test_shared_feature_rows': int(pd.util.hash_pandas_object(train[features], index=False).isin(pd.util.hash_pandas_object(test[features], index=False)).sum()),
    'numeric_target_correlations': train.select_dtypes(include=['number', 'bool']).corr()['satisfaction'].drop(['id','satisfaction'], errors='ignore').sort_values(ascending=False).to_dict() if pd.api.types.is_numeric_dtype(train.satisfaction) else {},
    'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')},
    'schema_checks': 'passed',
}
(ROOT/'reports/data-profile.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
