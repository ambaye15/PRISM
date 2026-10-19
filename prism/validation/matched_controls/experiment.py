from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .compensation import evaluate
from .eulerian import LEVELS, T, matched, score
from .storage import manifest, prepare, read_seed, seed_path, selected_seeds, write_csv


def _eulerian_seed(job):
    root, config, seed, force, save_grams = job
    if seed_path(root, 'eulerian', seed).exists() and not force:
        return read_seed(root, config, 'eulerian', seed)
    cases = [matched(seed, epsilon) for epsilon in config['levels']]
    rows = []
    for row, gram in cases:
        if save_grams:
            cache = root / 'grams'
            cache.mkdir(exist_ok=True)
            np.savez_compressed(cache / ('seed_' + str(seed) + '_eps_' + format(row['epsilon'], 'g') + '.npz'), gram=gram, times=T)
        rows.append(score(row, gram))
    frame = pd.DataFrame(rows)
    write_csv(frame, seed_path(root, 'eulerian', seed))
    return read_seed(root, config, 'eulerian', seed)


def _compensation_seed(job):
    root, config, seed, force = job
    if seed_path(root, 'compensation', seed).exists() and not force:
        return read_seed(root, config, 'compensation', seed)
    source = read_seed(root, config, 'eulerian', seed)
    rows = [evaluate(row) for row in source[source.epsilon > 0].to_dict('records')]
    write_csv(pd.DataFrame(rows), seed_path(root, 'compensation', seed))
    return read_seed(root, config, 'compensation', seed)


def _jobs(function, jobs, workers):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('Workers must be a positive integer')
    if not jobs:
        return pd.DataFrame()
    if workers == 1:
        frames = [function(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(function, jobs))
    return pd.concat(frames, ignore_index=True)


def run_eulerian(results_dir, seeds=tuple(range(8)), levels=LEVELS, workers=1,
                 shard=0, shards=1, force=False, save_grams=False):
    root, config = prepare(results_dir, seeds, levels)
    selected = selected_seeds(config, shard, shards)
    return _jobs(_eulerian_seed, [(root, config, seed, force, save_grams) for seed in selected], workers)


def run_compensation(results_dir, workers=1, shard=0, shards=1, force=False):
    root, config = manifest(results_dir)
    selected = selected_seeds(config, shard, shards)
    return _jobs(_compensation_seed, [(root, config, seed, force) for seed in selected], workers)


def run_experiment(results_dir, seeds=tuple(range(8)), levels=LEVELS, workers=1,
                   force=False, save_grams=False, plots=True):
    from .report import summarize
    run_eulerian(results_dir, seeds, levels, workers, force=force, save_grams=save_grams)
    run_compensation(results_dir, workers, force=force)
    return summarize(results_dir, plots=plots)
