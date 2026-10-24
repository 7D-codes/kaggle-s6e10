"""Frozen, conditional public-context experiment; no searches or submissions.

Run only under the root owner's separately authorized CPU experiment contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import time
import warnings

# The experiment fixes its CPU allocation before importing numerical libraries.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '4'

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.stats import rankdata
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

TRAIN_ROWS = 699635
TEST_ROWS = 299844
MEMBER_NAMES = ('tfm_causilo', 'tfm_exaone', 'tfm_kumolarge', 'tfm_kumosmall', 'tfm_limix2',
                'tfm_mitra2', 'tfm_tabfm', 'tfm_tabicl2', 'tfm_tabpfn35', 'own_realmlp_fixed3')
CONFIG = dict(C=1.0, solver='lbfgs', max_iter=2000, tol=1e-7, fit_intercept=True)
EXPERIMENT = dict(logistic=CONFIG, fold_seed=42, fold_count=5, meta_split_seed=20261025,
                  bootstrap_replicates=200, bootstrap_seed=314159, cpu_seconds=10800,
                  threads=4, uncertainty_reserve_seconds=600, correction_weight=.05,
                  replay_rtol=1e-6, replay_atol=1e-7, minimum_gain=.00005,
                  minimum_positive_folds=4, minimum_positive_cells=8,
                  features='logit clip1e-15 then clip[-30,30]; X,personal,personal*(X-rowmeanX)')


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vector_hash(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def probability(value, rows, name):
    value = np.asarray(value)
    if (value.shape != (rows,) or value.dtype.kind not in 'fiu' or
            not np.isfinite(value).all() or (value < 0).any() or (value > 1).any() or
            np.ptp(value) <= 1e-6):
        raise ValueError(f'{name}: expected finite nonconstant probability vector ({rows},)')
    return value.astype(np.float64)


def original_folds(y):
    folds = np.full(len(y), -1, dtype=np.int8)
    for k, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        folds[valid] = k
    return folds


def cells(y, folds):
    for k in range(5):
        rows = np.flatnonzero(folds == k)
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=20261025 + k).split(rows, y[rows]))
        for half, (train, valid) in enumerate(((a, b), (b, a))):
            yield k, half, rows[train], rows[valid]


def features(predictions, travel, contextual):
    travel = np.asarray(travel)
    if not np.isin(travel, ['Personal Travel', 'Business travel']).all():
        raise ValueError('Type of Travel contains unknown/missing values')
    x = np.clip(logit(np.clip(predictions, 1e-15, 1 - 1e-15)), -30, 30)
    personal = (travel == 'Personal Travel').astype(float)[:, None]
    return np.column_stack([x, personal, personal * (x - x.mean(axis=1, keepdims=True))]) if contextual else x


def checked_path(root, manifest, key):
    path = (root / manifest[key]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'{key}: input path escapes project root')
    if sha256(path) != manifest[key + '_sha256']:
        raise ValueError(f'{key}: SHA256 mismatch')
    return path


def exact_column(frame, column, expected):
    if column not in frame or not np.array_equal(frame[column].to_numpy(), expected):
        raise ValueError(f'{column}: row identity, target or fold mismatch')


def load_inputs(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if any(m.get(k) != v for k, v in dict(version=1, train_rows=TRAIN_ROWS, test_rows=TEST_ROWS,
            fold_seed=42, fold_count=5, meta_split_seed=20261025).items()):
        raise ValueError('manifest frozen dimensions/version/seeds mismatch')
    members = m['members']
    if tuple(member['name'] for member in members) != MEMBER_NAMES or not m.get('source_assurance'):
        raise ValueError('manifest frozen members/source assurance missing')
    train = pd.read_csv(checked_path(root, m, 'train'), usecols=['id', 'satisfaction', 'Type of Travel'])
    test = pd.read_csv(checked_path(root, m, 'test'), usecols=['id', 'Type of Travel'])
    if len(train) != TRAIN_ROWS or len(test) != TEST_ROWS:
        raise ValueError('official row counts mismatch')
    for frame in (train, test):
        if frame.id.dtype.kind not in 'iu' or frame.id.isna().any() or not frame.id.is_unique:
            raise ValueError('official IDs must be unique integers')
        if not frame['Type of Travel'].isin(['Personal Travel', 'Business travel']).all():
            raise ValueError('Type of Travel contains unknown/missing values')
    if np.intersect1d(train.id, test.id).size or not train.satisfaction.isin([0, 1]).all() or train.satisfaction.nunique() != 2:
        raise ValueError('official ID overlap or invalid target')
    y = train.satisfaction.to_numpy(dtype=np.uint8)
    folds = original_folds(y)
    for key, value, dtype in [('train_id', train.id, '<i8'), ('test_id', test.id, '<i8'),
                              ('target', y, 'uint8'), ('fold', folds, 'int8')]:
        if vector_hash(value, dtype) != m[key + '_sha256']:
            raise ValueError(f'{key}: canonical vector SHA256 mismatch')
    baseline_path = checked_path(root, m, 'baseline')
    baseline = pd.read_csv(baseline_path)
    exact_column(baseline, 'id', test.id)
    probability(baseline.satisfaction, TEST_ROWS, 'baseline')
    oof, test_predictions = [], []
    for i, member in enumerate(members):
        if (member.get('format') != ('csv' if i == 9 else 'npy') or member.get('representation') != 'probability' or
                member.get('fold_seed') != 42 or member.get('fold_count') != 5 or
                not member.get('provenance') or not member.get('source')):
            raise ValueError('member format/representation/folds/provenance mismatch')
        pair = []
        for key, count in [('oof', TRAIN_ROWS), ('test', TEST_ROWS)]:
            path = checked_path(root, member, key)
            if member['format'] == 'npy':
                value = np.load(path, allow_pickle=False)
            else:
                frame = pd.read_csv(path)
                exact_column(frame, 'id', train.id if key == 'oof' else test.id)
                if key == 'oof':
                    exact_column(frame, 'target', y)
                    exact_column(frame, 'fold', folds)
                value = frame['prediction' if key == 'oof' else 'satisfaction'].to_numpy()
            pair.append(probability(value, count, member['name'] + '/' + key))
        oof.append(pair[0]); test_predictions.append(pair[1])
    return dict(manifest=m, train=train, test=test, y=y, folds=folds, baseline=baseline,
                baseline_path=baseline_path, oof=np.column_stack(oof), test_predictions=np.column_stack(test_predictions))


def fit_model(x, y):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('fit requires both target classes')
    model = LogisticRegression(**CONFIG)
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('nonfinite fitted coefficients')
    return model, dict(coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]), n_iter=int(model.n_iter_[0]))


def replay(model, x):
    with threadpool_limits(limits=4):
        native = model.predict_proba(x)
    if native.shape != (len(x), 2) or not np.isfinite(native).all() or not np.allclose(native.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('native probabilities are not finite/normalized')
    prediction = probability(native[:, 1], len(x), 'native replay')
    manual = expit(x @ model.coef_[0] + model.intercept_[0])
    np.testing.assert_allclose(prediction, manual, rtol=1e-6, atol=1e-7)
    return prediction


def auc_record(y, control, context):
    a, b = float(roc_auc_score(y, control)), float(roc_auc_score(y, context))
    positives = int(np.count_nonzero(y))
    return dict(control=a, candidate=b, gain=b - a, rows=len(y), pairs=positives * (len(y) - positives))


def weighted(records):
    weights = [record['pairs'] for record in records]
    return {key: float(np.average([record[key] for record in records], weights=weights)) for key in ('control', 'candidate', 'gain')}


def summarize(y, control, context, cell):
    records = [dict(cell=int(k), fold=int(k // 2), **auc_record(y[cell == k], control[cell == k], context[cell == k])) for k in np.unique(cell)]
    folds = [dict(fold=int(k), **weighted([r for r in records if r['fold'] == k])) for k in sorted({r['fold'] for r in records})]
    leave_one = [dict(omitted_fold=int(k), **weighted([r for r in records if r['fold'] != k])) for k in sorted({r['fold'] for r in records})] if len(folds) > 1 else []
    return dict(cells=records, folds=folds, primary=weighted(records), pooled=auc_record(y, control, context), leave_one_original_fold_out=leave_one)


def bootstrap_gain(y, control, context, cell):
    rng = np.random.default_rng(314159)
    groups = [(np.flatnonzero((cell == k) & (y == 0)), np.flatnonzero((cell == k) & (y == 1))) for k in np.unique(cell)]
    weights = [len(a) * len(b) for a, b in groups]
    gains = []
    for _ in range(200):
        values = []
        for a, b in groups:
            rows = np.r_[rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True)]
            values.append(roc_auc_score(y[rows], context[rows]) - roc_auc_score(y[rows], control[rows]))
        gains.append(float(np.average(values, weights=weights)))
    return dict(label='conditional_fixed_prediction_diagnostic', replicates=200, seed=314159,
                interval95=np.percentile(gains, [2.5, 97.5]).tolist(), fixed_cell_pair_weights=weights,
                caveat='Omits training/selection variation and cross-cell dependence; no full-pipeline confidence interval.')


def screening_gate(metrics):
    return bool(metrics['valid'] and metrics['primary']['gain'] >= .00005 and
                sum(r['gain'] > 0 for r in metrics['folds']) >= 4 and
                sum(r['gain'] > 0 for r in metrics['cells']) >= 8 and metrics['bootstrap']['interval95'][0] > 0)


def check_projection(first_fold_fit_seconds, elapsed_seconds):
    projected = elapsed_seconds + 4 * first_fold_fit_seconds + 600
    if projected > 10800:
        raise TimeoutError(f'first-fold runtime projection plus 600s uncertainty reserve exceeds cap: {projected:.1f}s')
    return projected


def run(root, manifest_path, output):
    root, manifest_path, output = Path(root), Path(manifest_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metrics = dict(status='running', fit_count=0, config=EXPERIMENT, configuration_sha256=json_hash(EXPERIMENT),
                   source_sha256=sha256(Path(__file__)),
                   runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                scipy=scipy.__version__, sklearn=sklearn.__version__), progress=[])
    metrics['runtime_sha256'] = json_hash(metrics['runtime'])
    write_json(output / 'metrics.json', metrics)
    models, coefficients = [], []
    def receipt():
        metrics['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'metrics.json', metrics)
    try:
        metrics['manifest_sha256'] = sha256(manifest_path)
        data = load_inputs(root, manifest_path)
        m, train, test, y, folds = (data[k] for k in ('manifest', 'train', 'test', 'y', 'folds'))
        metrics.update(source_assurance=m['source_assurance'], input_manifest=m,
                       assurance_caveat='Conditional producer attestation; not fully certified pristine validation or exact-Summit proof.',
                       inference_caveat='Averaged base-test inputs differ from unavailable per-base-fold test inputs.')
        x = [features(data['oof'], train['Type of Travel'], contextual=a) for a in (False, True)]
        predictions = np.full((len(y), 2), np.nan)
        cell_id, coverage = np.full(len(y), -1, dtype=np.int8), np.zeros(len(y), dtype=np.int8)
        fold_fit_seconds = 0.
        for fold, half, fit_rows, valid_rows in cells(y, folds):
            if np.intersect1d(fit_rows, valid_rows).size or not np.all(folds[fit_rows] == fold) or not np.all(folds[valid_rows] == fold):
                raise ValueError('same-original-fold split isolation failed')
            for arm in range(2):
                fit_start = time.monotonic()
                model, coef = fit_model(x[arm][fit_rows], y[fit_rows])
                elapsed = time.monotonic() - fit_start
                if fold == 0: fold_fit_seconds += elapsed
                predictions[valid_rows, arm] = replay(model, x[arm][valid_rows])
                models.append((arm, model))
                coefficients.append(dict(fold=fold, half=half, arm=['control', 'context'][arm],
                    fit_rows_sha256=vector_hash(fit_rows, '<i8'), validation_rows_sha256=vector_hash(valid_rows, '<i8'),
                    fit_seconds=elapsed, replay_passed=True, **coef))
                metrics['fit_count'] += 1
                write_json(output / 'coefficients.json', coefficients)
                receipt()
            coverage[valid_rows] += 1; cell_id[valid_rows] = 2 * fold + half
            progress = dict(fold=fold, half=half, **auc_record(y[valid_rows], *predictions[valid_rows].T))
            metrics['progress'].append(progress)
            if half == 1:
                metrics.setdefault('fold_progress', []).append(dict(fold=fold, **weighted(metrics['progress'][-2:])))
                if fold == 0:
                    metrics['projected_seconds_including_reserve'] = check_projection(fold_fit_seconds, time.monotonic() - started)
            receipt()
            print(json.dumps(progress), flush=True)
        if metrics['fit_count'] != 20 or not np.all(coverage == 1) or set(cell_id) != set(range(10)) or not np.isfinite(predictions).all():
            raise ValueError('incomplete OOF cell coverage')
        pd.DataFrame(dict(id=train.id, target=y, fold=folds, cell=cell_id, control=predictions[:, 0], context=predictions[:, 1])).to_csv(output / 'oof.csv', index=False)
        metrics.update(summarize(y, *predictions.T, cell_id))
        metrics['bootstrap'] = bootstrap_gain(y, *predictions.T, cell_id)
        metrics['valid'] = True
        metrics['gate_passed'] = screening_gate(metrics)
        if metrics['gate_passed']:
            test_x = [features(data['test_predictions'], test['Type of Travel'], contextual=a) for a in (False, True)]
            test_prediction = np.zeros((len(test), 2))
            for arm, model in models:
                test_prediction[:, arm] += replay(model, test_x[arm]) / 10
            for arm in range(2): probability(test_prediction[:, arm], TEST_ROWS, 'averaged test')
            pd.DataFrame(dict(id=test.id, control=test_prediction[:, 0], context=test_prediction[:, 1])).to_csv(output / 'test.csv', index=False)
            baseline_rank = (rankdata(data['baseline'].satisfaction, method='average') - .5) / len(test)
            score = baseline_rank + .05 * (test_prediction[:, 1] - test_prediction[:, 0])
            candidate = pd.DataFrame(dict(id=test.id, satisfaction=(rankdata(score, method='average') - .5) / len(test)))
            exact_column(candidate, 'id', test.id)
            candidate.to_csv(output / 'summit-candidate.csv', index=False)
            shutil.copyfile(data['baseline_path'], output / 'summit-baseline.csv')
            if sha256(output / 'summit-baseline.csv') != m['baseline_sha256']:
                raise ValueError('archived Summit baseline hash mismatch')
        metrics['artifacts'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.name != 'metrics.json'}
        metrics.update(status='complete', decision='conditional development gate passed; unsubmitted/unscored against Summit' if metrics['gate_passed'] else 'rejected/inconclusive; no Summit candidate exported')
        receipt()
        return metrics
    except BaseException as error:
        metrics.update(status='failed', error=f'{type(error).__name__}: {error}')
        receipt()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    def timeout(signum, frame):
        raise TimeoutError('frozen 10800-second CPU run deadline exceeded')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 10800)
    try:
        run(args.root, args.manifest, args.output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    main()
