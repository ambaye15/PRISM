import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config import BEHAVIORS, LOWER, METRICS, N_WINDOWS, PARAMETERS, UPPER
from .config import design, manifest, scales_for, signature, workspace
from .experiment import batch, run_decode, summarize
from .model import PortableDecoder, train_decoder


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def expanded_design(training_samples, sampler_seed=20261019):
    require(isinstance(training_samples, int) and training_samples >= 144
            and training_samples % 3 == 0,
            'Training sizes must be multiples of three, at least 144')
    require(isinstance(sampler_seed, int) and 0 <= sampler_seed < 2**32 - 3,
            'Sampler seed must be an integer between 0 and 2**32 - 4')
    rows = design()
    extra = (training_samples - 144) // 3
    if not extra:
        return rows
    test = pd.DataFrame([row for row in rows if row['split'] == 'test'])
    forbidden = [test[key].unique() for key in PARAMETERS]
    for behavior_index, behavior in enumerate(BEHAVIORS):
        exponent = math.ceil(math.log2(extra + 64))
        while True:
            unit = qmc.Sobol(d=3, scramble=True, seed=sampler_seed + behavior_index)
            values = LOWER + unit.random_base2(exponent) * (UPPER - LOWER)
            keep = np.ones(len(values), dtype=bool)
            for axis, held_out in enumerate(forbidden):
                keep &= ~np.isclose(values[:, axis, None], held_out[None, :],
                                    atol=1e-10, rtol=0).any(axis=1)
            values = values[keep]
            if len(values) >= extra:
                break
            exponent += 1
        for index, point in enumerate(values[:extra]):
            rows.append(dict(
                condition_id=f'{behavior}__library_extra__{index:06d}',
                behavior=behavior, split='library',
                seed=100000 + 100000 * behavior_index + index,
                **dict(zip(PARAMETERS, point.tolist()))))
    require(len({row['seed'] for row in rows}) == len(rows), 'Sampler seed collision')
    return rows


def check_data(data, expected):
    columns = ['condition_id', 'behavior', 'split', 'seed', 'status',
               'image_sha256', 'windows_retained', *METRICS, *PARAMETERS]
    missing = sorted(set(columns) - set(data))
    require(not missing, 'Incomplete measurement columns: ' + ', '.join(missing))
    require(data.condition_id.is_unique, 'Duplicate measured condition IDs')
    expected = pd.DataFrame(expected).set_index('condition_id')
    require(set(data.condition_id) == set(expected.index), 'Measured design differs')
    indexed = data.set_index('condition_id').loc[expected.index]
    for key in ['behavior', 'split', 'seed']:
        require(indexed[key].eq(expected[key]).all(), 'Design differs: ' + key)
    require(np.allclose(indexed[PARAMETERS], expected[PARAMETERS], atol=1e-12, rtol=0),
            'Measured simulator parameters differ from the saved design')
    failed = data.loc[~data.status.eq('ok'), 'condition_id'].tolist()
    require(not failed, 'Unavailable measurements; inspect records: ' + ', '.join(failed[:6]))
    require(np.isfinite(data[METRICS + PARAMETERS].to_numpy(float)).all(),
            'Non-finite measured profiles')
    require(data.windows_retained.eq(N_WINDOWS).all(), 'Unexpected window support')
    require(data.image_sha256.notna().all(), 'Missing image hashes')
    train, test = data[data.split.eq('library')], data[data.split.eq('test')]
    require(not set(train.seed) & set(test.seed), 'Training/test seed overlap')
    require(not set(train.image_sha256) & set(test.image_sha256), 'Training/test image overlap')
    for key in PARAMETERS:
        require(not np.isclose(train[key].to_numpy()[:, None], test[key].unique()[None, :],
                               atol=1e-10, rtol=0).any(), 'Held-out axis overlaps: ' + key)
    require(len(test) == 54, 'The original 54 held-out targets must remain fixed')
    return dict(training_samples=len(train), test_samples=len(test),
                held_out_axes_checked=True, image_leakage_checked=True)


def reuse_records(source, target, rows, pipeline_signature):
    if source is None:
        return 0
    source, target = Path(source).resolve(), Path(target).resolve()
    require(source != target, 'Reuse source must differ from the new study directory')
    require((source / 'records').is_dir(), 'Reuse source has no records directory: ' + str(source))
    count = 0
    (target / 'records').mkdir(exist_ok=True)
    for row in rows:
        name = row['condition_id'] + '.json'
        path = source / 'records' / name
        if not path.is_file() or (target / 'records' / name).exists():
            continue
        try:
            record = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        row_hash = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if (record.get('pipeline_signature') == pipeline_signature
                and record.get('input_row_sha256') == row_hash
                and record.get('status') == 'ok'):
            shutil.copy2(path, target / 'records' / name)
            count += 1
    return count


def fixed_retrieval(data, scales):
    library = data[data.split.eq('library')].copy()
    test = data[data.split.eq('test')].sort_values('condition_id')
    output = []
    rng = np.random.default_rng(20261017)
    for query in test.to_dict('records'):
        group = library[library.behavior.eq(query['behavior'])].sort_values('condition_id')
        target = np.asarray([query[key] for key in METRICS], dtype=float)
        scale = np.asarray(scales[query['behavior']], dtype=float)
        absolute = np.abs(group[METRICS].to_numpy(float) - target)
        delta = absolute / scale
        distance = np.sqrt(np.mean(delta**2, axis=1))
        selections = [('retrieval', int(np.argmin(distance))),
                      ('without_ps', int(np.argmin(np.mean(delta[:, :3]**2, axis=1))))]
        for method, index in selections:
            output.append(dict(query_id=query['condition_id'], behavior=query['behavior'],
                method=method, candidate_id=group.iloc[index].condition_id,
                profile_error=float(distance[index]),
                **{key + '_abs_error': float(absolute[index, j]) for j, key in enumerate(METRICS)},
                **{key + '_scaled_error': float(delta[index, j]) for j, key in enumerate(METRICS)}))
        selected = rng.integers(0, len(group), size=100)
        output.append(dict(query_id=query['condition_id'], behavior=query['behavior'],
            method='random_100_mean', candidate_id='', profile_error=float(distance[selected].mean()),
            **{key + '_abs_error': float(absolute[selected, j].mean()) for j, key in enumerate(METRICS)},
            **{key + '_scaled_error': float(delta[selected, j].mean()) for j, key in enumerate(METRICS)}))
    return pd.DataFrame(output)


def validate_model(root, training_samples, sampler_seed, scales):
    root = Path(root)
    data = pd.read_csv(root / 'library_and_test.csv')
    report = check_data(data, expanded_design(training_samples, sampler_seed))
    decoder = PortableDecoder.load(root)
    library, test = data[data.split.eq('library')], data[data.split.eq('test')]
    require(set(decoder.metadata['training_ids']) == set(library.condition_id),
            'Decoder training IDs differ from the requested subset')
    require(decoder.metadata['training_rows'] == training_samples, 'Wrong training sample count')
    require(decoder.metadata['hidden_layers'] == [16, 16]
            and decoder.metadata['trainable_scalars'] == 451, 'Decoder architecture changed')
    decoded = pd.read_csv(root / 'decoded.csv')
    require(len(decoded) == 54 and decoded.query_id.is_unique
            and set(decoded.query_id) == set(test.condition_id), 'Decoded target coverage differs')
    require(decoded.status.eq('ok').all(), 'Some neural outputs failed; inspect model records')
    require(np.isfinite(decoded[METRICS].to_numpy(float)).all()
            and decoded.windows_retained.eq(N_WINDOWS).all(), 'Invalid decoded profiles')
    require(not set(decoded.seed) & set(data.seed), 'Decoded/input seed overlap')
    require(not set(decoded.image_sha256) & set(data.image_sha256), 'Decoded/input image overlap')
    require(np.isfinite(decoded[PARAMETERS].to_numpy(float)).all()
            and (decoded[PARAMETERS].to_numpy(float) >= LOWER).all()
            and (decoded[PARAMETERS].to_numpy(float) <= UPPER).all(), 'Invalid decoded parameters')
    saved_scales = json.loads((root / 'scales.json').read_text())
    for behavior in BEHAVIORS:
        require(np.allclose(saved_scales[behavior], scales[behavior], atol=1e-12, rtol=0),
                'Profile scales changed between training sizes')
    evaluation = pd.read_csv(root / 'evaluation.csv')
    require(len(evaluation) == 216 and not evaluation.duplicated(['query_id', 'method']).any(),
            'Incomplete or duplicate evaluations')
    methods = {'neural_decoder', 'retrieval', 'without_ps', 'random_100_mean'}
    require(set(evaluation.method) == methods
            and set(evaluation.query_id) == set(test.condition_id), 'Evaluation coverage differs')
    expected_baselines = fixed_retrieval(data, scales).sort_values(['query_id', 'method'])
    actual_baselines = evaluation[~evaluation.method.eq('neural_decoder')].sort_values(['query_id', 'method'])
    require(actual_baselines[['query_id', 'method']].reset_index(drop=True).equals(
            expected_baselines[['query_id', 'method']].reset_index(drop=True)), 'Baseline keys differ')
    score_columns = ['profile_error', *[key + '_abs_error' for key in METRICS],
                     *[key + '_scaled_error' for key in METRICS]]
    require(np.allclose(actual_baselines[score_columns], expected_baselines[score_columns],
                        atol=1e-11, rtol=1e-11), 'Baseline distances or ranking differ')
    require(actual_baselines.candidate_id.fillna('').tolist()
            == expected_baselines.candidate_id.fillna('').tolist(), 'Baseline candidate IDs differ')
    targets = test.set_index('condition_id')
    neural = evaluation[evaluation.method.eq('neural_decoder')].set_index('query_id')
    for row in decoded.to_dict('records'):
        delta = np.abs(np.asarray([row[key] for key in METRICS])
                       - targets.loc[row['query_id'], METRICS].to_numpy(float))
        scaled = delta / np.asarray(scales[row['behavior']])
        measured = neural.loc[row['query_id']]
        require(np.isclose(measured.profile_error, np.sqrt(np.mean(scaled**2)),
                           atol=1e-11, rtol=1e-11), 'Neural output was not correctly remeasured')
        require(np.allclose(measured[[key + '_abs_error' for key in METRICS]].to_numpy(float),
                            delta, atol=1e-11, rtol=1e-11), 'Neural coordinate errors differ')
    report.update(neural_outputs_remeasured=54, evaluation_rows=216,
                  training_rows=decoder.metadata['training_rows'], trainable_scalars=451,
                  profile_scales_fixed=True, pipeline_signature=manifest(root)['pipeline_signature'])
    return report


def collect(root, protocol):
    rows = []
    for size in protocol['training_sizes']:
        directory = root / f'train_{size}'
        evaluation = pd.read_csv(directory / 'evaluation.csv')
        training = json.loads((directory / 'decoder_training.json').read_text())
        decoded = pd.read_csv(directory / 'decoded.csv')
        clipped = decoded.parameter_clipped.astype(str).str.lower().eq('true').sum()
        geometry = evaluation.groupby(['behavior', 'method']).agg(
            n=('profile_error', 'size'), median_profile_error=('profile_error', 'median'),
            q90_profile_error=('profile_error', lambda values: values.quantile(.9)),
            frequency_mae=('frequency_abs_error', 'mean'), ps_mae=('ps_abs_error', 'mean'))
        geometry.to_csv(directory / 'geometry_summary.csv')
        write_json(directory / 'sample_size_decoder_stats.json', dict(
            training_rows=size, trainable_scalars=training['trainable_scalars'],
            optimizer_iterations=training['n_iter'], parameter_clipped=int(clipped),
            optimizer_reached_iteration_limit=training['n_iter'] >= training['max_iter'],
            decoder_sha256=hashlib.sha256((directory / 'decoder.npz').read_bytes()).hexdigest(),
            scoring='Fixed original 144-library IQR scales'))
        for method, group in evaluation.groupby('method'):
            rows.append(dict(training_samples=size, method=method, n=len(group),
                median_profile_error=float(group.profile_error.median()),
                q90_profile_error=float(group.profile_error.quantile(.9)),
                within_035=int(group.profile_error.le(.35).sum()),
                **{key + '_mae': float(group[key + '_abs_error'].mean()) for key in METRICS},
                optimizer_iterations=training['n_iter'], parameter_clipped=int(clipped),
                optimizer_reached_iteration_limit=training['n_iter'] >= training['max_iter']))
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'summary.csv', index=False)
    frame.to_json(root / 'summary.json', orient='records', indent=2)
    return frame


def run_study(results_dir, training_sizes=(144, 576, 1440), workers=4,
              sampler_seed=20261019, reuse_results=None, force=False,
              make_plots=True, make_video=False):
    require(workers >= 1, 'Workers must be positive')
    sizes = sorted(set(training_sizes))
    require(sizes and sizes[0] == 144, 'Include the original 144-sample baseline')
    for size in sizes:
        expanded_design(size, sampler_seed)
    root = Path(results_dir).resolve()
    require(not root.is_relative_to(Path(__file__).resolve().parents[1]),
            'Choose a results directory outside the installed prism package')
    measurement_signature = signature(1)
    source = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ['sample_size.py', 'model.py', 'experiment.py']}
    protocol = dict(schema_version=1, study='exemplar_training_sample_size',
        training_sizes=sizes, test_samples=54, sampler_seed=sampler_seed,
        flow_lag_profile_power=1, pipeline_signature=measurement_signature,
        hidden_layers=[16, 16], trainable_scalars=451, source_sha256=source,
        scoring='Original 144-library IQR scales, fixed across all sizes and methods',
        sampler='Original library plus nested scrambled Sobol parameters; original tests excluded')
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'sample_size_study.json'
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text()) == protocol,
                'Study settings or code changed; use a new results directory')
    else:
        require(not any(root.iterdir()), 'Use an empty or compatible study directory')
        write_json(protocol_path, protocol)
    rows = expanded_design(sizes[-1], sampler_seed)
    write_json(root / 'design.json', rows)
    reused = reuse_records(reuse_results, root, rows, measurement_signature)
    print(f'Reused {reused} compatible records; measuring {len(rows)} shared input conditions', flush=True)
    data = batch(rows, root, 1, workers, force)
    check_data(data, rows)
    data.to_csv(root / 'library_and_test.csv', index=False)
    baseline_ids = {row['condition_id'] for row in design() if row['split'] == 'library'}
    scales = {key: value.tolist() for key, value in scales_for(
        data[data.condition_id.isin(baseline_ids)]).items()}
    write_json(root / 'reference_scales.json', scales)
    reports = {}
    for size in sizes:
        directory, metadata = workspace(root / f'train_{size}', 1)
        selected = {row['condition_id'] for row in expanded_design(size, sampler_seed)}
        subset = data[data.condition_id.isin(selected)].copy().sort_values('condition_id')
        subset.to_csv(directory / 'library_and_test.csv', index=False)
        metadata.update(pipeline_signature=measurement_signature, training_samples=size,
                        sample_size_study=str(root), fixed_profile_scales=scales)
        write_json(directory / 'experiment.json', metadata)
        retrain = force or not (directory / 'decoder.npz').is_file()
        if not retrain:
            try:
                PortableDecoder.load(directory)
            except (ValueError, KeyError, OSError):
                retrain = True
        if retrain:
            train_decoder(directory)
        decoded = run_decode(directory, workers=workers, force=force)
        fixed_retrieval(subset, scales).to_csv(directory / 'retrieval_evaluation.csv', index=False)
        write_json(directory / 'scales.json', scales)
        summarize(subset, decoded, directory)
        reports[str(size)] = validate_model(directory, size, sampler_seed, scales)
        write_json(directory / 'sample_size_validation.json', reports[str(size)])
        if make_video:
            from .illustrate import illustrate
            illustrate(directory)
    frame = collect(root, protocol)
    write_json(root / 'validation.json', reports)
    if make_plots:
        from prism.plotting.exemplar_sample_size import plot_sample_size
        plot_sample_size(frame, root)
    print(frame[frame.method.eq('neural_decoder')].to_string(index=False), flush=True)
    return dict(results_dir=str(root), training_sizes=sizes, test_samples=54,
                profile_scales_fixed=True, trainable_scalars=451,
                complete_training_samples=sizes[-1], summary=str(root / 'summary.csv'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare more training samples with the unchanged exemplar decoder')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--results-dir', type=Path, default=Path('runs/exemplar_sample_size'))
    run.add_argument('--training-sizes', type=int, nargs='+', default=[144, 576, 1440])
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--sampler-seed', type=int, default=20261019)
    run.add_argument('--reuse-results', type=Path)
    run.add_argument('--force', action='store_true')
    run.add_argument('--no-plots', action='store_true')
    run.add_argument('--illustrate', action='store_true')
    validation = commands.add_parser('validate')
    validation.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = run_study(args.results_dir, args.training_sizes, args.workers,
                               args.sampler_seed, args.reuse_results, args.force,
                               not args.no_plots, args.illustrate)
        else:
            root = args.results_dir.resolve()
            protocol = json.loads((root / 'sample_size_study.json').read_text())
            scales = json.loads((root / 'reference_scales.json').read_text())
            result = {str(size): validate_model(root / f'train_{size}', size,
                                               protocol['sampler_seed'], scales)
                      for size in protocol['training_sizes']}
            write_json(root / 'validation.json', result)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    main()
