"""Frozen three-arm Kumo comparison. Explicit run only; never submits."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import signal
import time

# Import first: the reviewed helper fixes all numerical thread allocations at four.
import stack_public_context as m
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from train_baselines import load_teacher_features

CONTROL = Path('reports/summit-score-goal/kumo-continuation-control-manifest.json')
CONTROL_SHA = '91a6fac84d243482579d18ef5c2ab52b33b6ecf5ad22c22c457bcddf094d628d'
BASE11_SHA = '72ab4f3c6dd81f63cbb342f763bfbf01713e79e9a85a5b55c320cd5a00f3729e'
BASE10_SHA = '857531628a486634bb2d983f5cd6adc4b793a4ef8918f7348a4735c8bb155ccb'
HELPER_SHA = 'b20c9043c05eebe17d5673839312af6e12466db50577f303800ed16e15e010ec'
AUDIT = Path('reports/summit-score-goal/kumo-continuation-execution/independent-terminal-audit.json')
PAIRED = Path('results/summit-kumo-continuation-v1/kumo-pair-execution/paired')
PRIOR = Path('results/summit-catboost-comparison-v1')
OUTPUT = Path('results/summit-kumo-comparison-v1')
JOB = 'wasidyms/s6e10-codex-kumo-continuation-v1/1'
ARMS = ('control12', 'raw13', 'joint14')
CANDIDATE = 'candidate-kumo-UNSCORED.csv'
INCOMPLETE = 'candidate-kumo.INCOMPLETE.csv'
CONFIG = dict(logistic=m.CONFIG, fold_seed=42, half_seed=20261025, arms=list(ARMS),
    fits=30, cells=10, threads=4, cpu_seconds=10800, bootstrap_replicates=200,
    bootstrap_seed=314159, min_gain=.00002, min_positive_folds=4, correction_weight=.05)


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'input/output path escapes declared root: {name}')
    return path


def hashed(root, name, expected, hashes):
    path = inside(root, name)
    digest = m.sha256(path)
    if digest != expected:
        raise ValueError(f'{name}: SHA256 mismatch')
    hashes[str(path.relative_to(root.resolve()))] = digest
    return path


def frozen_manifest(root, name, expected, hashes):
    return json.loads(hashed(root, name, expected, hashes).read_text())


def qualified_audit(audit, job, kernel=None):
    if (any(audit.get(key) is not True for key in ('valid', 'complete_oof', 'complete_test'))
            or audit.get('job') != job or (kernel is not None and
            (audit.get('kernel_id') != kernel or audit.get('completed_folds') != list(range(5))))):
        raise ValueError('independent terminal/native audit is incomplete or has wrong job identity')


def load_inputs(root):
    """Qualify the frozen chain and independent terminal receipt before any fit."""
    root = Path(root).resolve(); hashes = {}
    control = frozen_manifest(root, CONTROL, CONTROL_SHA, hashes)
    if m.sha256(Path(m.__file__)) != HELPER_SHA or m.sha256(inside(root, 'stack_public_context.py')) != HELPER_SHA:
        raise ValueError('reviewed helper source SHA256 mismatch')
    base11 = frozen_manifest(root, control['base11_manifest'], BASE11_SHA, hashes)
    if control['base11_manifest_sha256'] != BASE11_SHA or base11['base_manifest_sha256'] != BASE10_SHA:
        raise ValueError('frozen manifest chain mismatch')
    base10 = hashed(root, base11['base_manifest'], BASE10_SHA, hashes)
    names = list(m.MEMBER_NAMES) + ['original_only_xgb_teacher']
    for manifest, expected in ((base11, names), (control, names + ['own_catboost_ctr'])):
        if (manifest.get('members') != expected or manifest.get('version') != 1 or
                manifest.get('meta_half_seed') != 20261025 or manifest.get('logistic_parameters') != m.CONFIG):
            raise ValueError('frozen members, parameters or seeds mismatch')
    data = m.load_inputs(root, base10)
    y, folds, train, test = (data[key] for key in ('y', 'folds', 'train', 'test'))
    for manifest in (base11, control):
        if manifest['fold_sha256'] != m.vector_hash(folds, 'int8'):
            raise ValueError('frozen fold hash mismatch')
    teacher_path = hashed(root, base11['teacher_csv'], base11['teacher_sha256'], hashes)
    teacher, teacher_test = load_teacher_features(teacher_path, train.id, test.id)
    teacher = m.probability(teacher, len(train), 'teacher OOF')
    teacher_test = m.probability(teacher_test, len(test), 'teacher test')
    cat = control['catboost']
    cat_audit_path = hashed(root, cat['native_audit'], cat['native_audit_sha256'], hashes)
    cat_audit = json.loads(cat_audit_path.read_text())
    qualified_audit(cat_audit, 'wasidyms/s6e10-codex-catboost-ctr-v1/1')
    cat_frames = []
    for key, ids in (('oof', train.id), ('test', test.id)):
        path = hashed(root, cat[key], cat[key + '_sha256'], hashes)
        if cat_audit['output_sha256'][key + '.csv'] != cat[key + '_sha256']:
            raise ValueError('Cat native audit output hash mismatch')
        frame = pd.read_csv(path); m.exact_column(frame, 'id', ids)
        if key == 'oof':
            m.exact_column(frame, 'satisfaction', y); m.exact_column(frame, 'fold', folds)
        cat_frames.append(m.probability(frame.catboost.to_numpy(), len(ids), 'Cat ' + key))
    audit_path = inside(root, AUDIT)
    audit = json.loads(audit_path.read_text())
    hashes[str(AUDIT)] = m.sha256(audit_path)
    qualified_audit(audit, JOB, 141062725)
    kumo = []
    for key, ids in (('oof', train.id), ('test', test.id)):
        path = hashed(root, PAIRED / (key + '.csv'), audit['output_sha256'][key + '.csv'], hashes)
        frame = pd.read_csv(path)
        expected = ['id', 'satisfaction', 'fold', 'raw', 'fe'] if key == 'oof' else ['id', 'raw', 'fe']
        if list(frame) != expected or len(frame) != len(ids):
            raise ValueError('Kumo columns or full row coverage mismatch')
        m.exact_column(frame, 'id', ids)
        if key == 'oof':
            m.exact_column(frame, 'satisfaction', y); m.exact_column(frame, 'fold', folds)
        kumo.append(np.column_stack([m.probability(frame[arm].to_numpy(), len(ids), 'Kumo ' + key + '/' + arm) for arm in ('raw', 'fe')]))
    baseline_path = hashed(root, control['baseline_candidate'], control['baseline_sha256'], hashes)
    baseline = pd.read_csv(baseline_path); m.exact_column(baseline, 'id', test.id)
    m.probability(baseline.satisfaction.to_numpy(), len(test), 'Cat candidate baseline')
    prior = []
    for key, ids in (('oof', train.id), ('test', test.id)):
        path = inside(root, PRIOR / (key + '.csv')); hashes[str(PRIOR / (key + '.csv'))] = m.sha256(path)
        frame = pd.read_csv(path); m.exact_column(frame, 'id', ids)
        if key == 'oof':
            m.exact_column(frame, 'target', y); m.exact_column(frame, 'fold', folds)
            expected_cell = np.full(len(y), -1, dtype=np.int8)
            for fold, half, _, valid in m.cells(y, folds): expected_cell[valid] = 2 * fold + half
            m.exact_column(frame, 'cell', expected_cell)
        m.probability(frame.candidate.to_numpy(), len(ids), 'retained control ' + key)
        prior.append(frame)
    # The ten-member loader has already verified each original input hash.
    hashes['base10_inputs'] = data['manifest']
    data.update(oof=np.column_stack([data['oof'], teacher, cat_frames[0], kumo[0]]),
        test_predictions=np.column_stack([data['test_predictions'], teacher_test, cat_frames[1], kumo[1]]),
        control=control, base11_manifest=base11, audit=audit, input_sha256=hashes,
        baseline=baseline, baseline_path=baseline_path, prior_oof=prior[0], prior_test=prior[1])
    return data


def paired_fit(y, folds, matrices, checkpoint=None):
    if len(matrices) != 3 or [x.shape for x in matrices] != [(len(y), k) for k in (12, 13, 14)]:
        raise ValueError('frozen three-arm feature dimensions mismatch')
    prediction = np.full((len(y), 3), np.nan); cell = np.full(len(y), -1, dtype=np.int8)
    coverage = np.zeros(len(y), dtype=np.int8); models, states = [], []
    for fold, half, fitting, validation in m.cells(y, folds):
        if (np.intersect1d(fitting, validation).size or not np.all(folds[fitting] == fold)
                or not np.all(folds[validation] == fold)):
            raise ValueError('same-original-fold split isolation failed')
        for arm, x in enumerate(matrices):
            started = time.process_time()
            model, state = m.fit_model(x[fitting], y[fitting])
            prediction[validation, arm] = m.replay(model, x[validation])
            models.append((arm, model))
            states.append(dict(arm=ARMS[arm], fold=fold, half=half,
                fit_rows_sha256=m.vector_hash(fitting, '<i8'), validation_rows_sha256=m.vector_hash(validation, '<i8'),
                fit_rows=len(fitting), validation_rows=len(validation), native_manual_replay_passed=True,
                cpu_seconds=time.process_time() - started,
                native_state=dict(format='sklearn_LogisticRegression_JSON', classes=model.classes_.tolist(),
                    n_features_in=int(model.n_features_in_), parameters=m.CONFIG), **state))
            if checkpoint: checkpoint(states)
        cell[validation] = 2 * fold + half; coverage[validation] += 1
    if len(models) != 30 or not np.all(coverage == 1) or set(cell) != set(range(10)) or not np.isfinite(prediction).all():
        raise ValueError('incomplete OOF cell coverage')
    return prediction, cell, models, states


def test_predictions(models, matrices):
    prediction = np.zeros((len(matrices[0]), 3)); count = np.zeros(3, int)
    for arm, model in models:
        prediction[:, arm] += m.replay(model, matrices[arm]) / 10
        count[arm] += 1
    if not np.array_equal(count, [10, 10, 10]): raise ValueError('incomplete ten-model test coverage')
    for arm in range(3): m.probability(prediction[:, arm], len(prediction), ARMS[arm] + ' averaged test')
    return prediction


# Prevent pytest treating the public prediction helper as a test on import.
test_predictions.__test__ = False


def promotion_gate(fe, joint, valid):
    return bool(valid and all(record['primary']['gain'] >= .00002 and
        sum(fold['gain'] > 0 for fold in record['folds']) >= 4 and
        record['bootstrap']['interval95'][0] > 0 for record in (fe, joint)))


def corrected_rank(baseline, control, joint):
    rank = (rankdata(baseline, method='average') - .5) / len(baseline)
    return (rankdata(rank + .05 * (np.asarray(joint) - np.asarray(control)), method='average') - .5) / len(baseline)


def artifact_hashes(output):
    return {path.name: m.sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != 'metrics.json'}


def quarantine(output, metrics):
    final, partial = output / CANDIDATE, output / INCOMPLETE
    if final.exists(): final.replace(partial)
    metrics['artifacts_sha256'] = artifact_hashes(output)


def run(root):
    root = Path(root).resolve(); output = inside(root, OUTPUT)
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    metrics = dict(status='running', valid=False, gate_passed=False, fit_count=0, submissions=0,
        public_score_improvement_verified=False, config=CONFIG, configuration_sha256=m.json_hash(CONFIG),
        source_sha256={path.name: m.sha256(path) for path in (Path(__file__), Path(m.__file__), Path(__file__).with_name('train_baselines.py'))},
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=m.sklearn.__version__, scipy=m.scipy.__version__),
        assurance_caveat='Conditional producer attestation and adaptive development; exact Summit OOF missing, no public-score proof.',
        inference_caveat='Average ten frozen models per arm on averaged base-test inputs; unavailable per-base-fold inputs prevent exact equivalent inference.')
    metrics['runtime_sha256'] = m.json_hash(metrics['runtime'])
    def save():
        metrics.update(elapsed_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
        m.write_json(output / 'metrics.json', metrics)
    def timeout(signum, frame): raise TimeoutError('frozen 10800-second CPU/wall comparison cap exceeded')
    previous = signal.signal(signal.SIGPROF, timeout); previous_wall = signal.signal(signal.SIGALRM, timeout)
    save()
    signal.setitimer(signal.ITIMER_PROF, 10800); signal.setitimer(signal.ITIMER_REAL, 10800)
    try:
        data = load_inputs(root)
        metrics.update(input_sha256=data['input_sha256'], control_manifest=data['control'],
            base11_manifest=data['base11_manifest'], source_assurance=data['manifest']['source_assurance'], independent_native_audit=data['audit'])
        save()
        x = [m.features(data['oof'][:, :k], data['train']['Type of Travel'], False) for k in (12, 13, 14)]
        def checkpoint(states):
            metrics['fit_count'] = len(states); m.write_json(output / 'coefficients.json', states); save()
        prediction, cell, models, states = paired_fit(data['y'], data['folds'], x, checkpoint)
        m.exact_column(data['prior_oof'], 'cell', cell)
        np.testing.assert_allclose(prediction[:, 0], data['prior_oof'].candidate, rtol=1e-6, atol=1e-7)
        metrics['control_oof_replay_max_abs_error'] = float(np.max(abs(prediction[:, 0] - data['prior_oof'].candidate)))
        pd.DataFrame(dict(id=data['train'].id, target=data['y'], fold=data['folds'], cell=cell,
            **dict(zip(ARMS, prediction.T)))).to_csv(output / 'oof.csv', index=False)
        metrics['coverage'] = dict(rows=len(cell), cells=10, fits=30, exactly_once=True,
            train_id_sha256=m.vector_hash(data['train'].id, '<i8'), test_id_sha256=m.vector_hash(data['test'].id, '<i8'),
            target_sha256=m.vector_hash(data['y'], 'uint8'), fold_sha256=m.vector_hash(data['folds'], 'int8'), cell_sha256=m.vector_hash(cell, 'int8'))
        xt = [m.features(data['test_predictions'][:, :k], data['test']['Type of Travel'], False) for k in (12, 13, 14)]
        final = test_predictions(models, xt)
        np.testing.assert_allclose(final[:, 0], data['prior_test'].candidate, rtol=1e-6, atol=1e-7)
        metrics['control_test_replay_max_abs_error'] = float(np.max(abs(final[:, 0] - data['prior_test'].candidate)))
        pd.DataFrame(dict(id=data['test'].id, **dict(zip(ARMS, final.T)))).to_csv(output / 'test.csv', index=False)
        for name, a, b in [('fe_14_vs_13', 1, 2), ('joint_14_vs_12', 0, 2), ('raw_13_vs_12_descriptive', 0, 1)]:
            summary = m.summarize(data['y'], prediction[:, a], prediction[:, b], cell)
            if 'descriptive' not in name: summary['bootstrap'] = m.bootstrap_gain(data['y'], prediction[:, a], prediction[:, b], cell)
            metrics[name] = summary; save()
        passed = promotion_gate(metrics['fe_14_vs_13'], metrics['joint_14_vs_12'], valid=True)
        if passed:
            candidate = pd.DataFrame(dict(id=data['test'].id, satisfaction=corrected_rank(data['baseline'].satisfaction, final[:, 0], final[:, 2])))
            m.probability(candidate.satisfaction.to_numpy(), m.TEST_ROWS, 'Kumo candidate')
            shutil.copyfile(data['baseline_path'], output / 'catboost-baseline.csv')
            expected = data['control']['baseline_sha256']
            if m.sha256(output / 'catboost-baseline.csv') != expected or m.sha256(data['baseline_path']) != expected:
                raise ValueError('preserved Cat baseline SHA256 mismatch')
            stage = output / INCOMPLETE; candidate.to_csv(stage, index=False)
            readback = pd.read_csv(stage); m.exact_column(readback, 'id', data['test'].id)
            m.probability(readback.satisfaction.to_numpy(), m.TEST_ROWS, 'candidate readback')
            np.testing.assert_allclose(readback.satisfaction, candidate.satisfaction, rtol=1e-6, atol=1e-7)
            stage.replace(output / CANDIDATE)
        metrics['artifacts_sha256'] = artifact_hashes(output)
        metrics.update(status='complete', valid=True, gate_passed=passed,
            decision='both conditional gates passed; candidate UNSCORED, unsubmitted' if passed else 'rejected/inconclusive; failed gate is not a precise null; no candidate exported')
        save(); return metrics
    except BaseException as error:
        metrics.update(status='failed', valid=False, gate_passed=False, decision='failed; no promoted candidate accepted', error=f'{type(error).__name__}: {error}')
        quarantine(output, metrics); save(); raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0); signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGPROF, previous); signal.signal(signal.SIGALRM, previous_wall)


def self_test():
    rng = np.random.default_rng(73); y = np.arange(400) % 2
    x = rng.normal(size=(400, 14)) + .3 * y[:, None]
    prediction, cell, models, states = paired_fit(y, m.original_folds(y), [x[:, :k] for k in (12, 13, 14)])
    assert len(models) == len(states) == 30 and set(cell) == set(range(10))
    assert all(np.ptp(prediction[:, k]) > 1e-6 for k in range(3))
    print('synthetic thirty-fit self-check passed; no competition fit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true'); mode.add_argument('--self-test', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    self_test() if args.self_test else run(args.root)


if __name__ == '__main__': main()
