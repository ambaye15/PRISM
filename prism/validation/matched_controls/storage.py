import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .eulerian import LEVELS


def output_root(path):
    root = Path(path).expanduser().resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Choose a results directory outside the prism source package')
    return root


def signature():
    package = Path(__file__).resolve().parents[2]
    paths = [*sorted(Path(__file__).parent.glob('*.py')),
             package / 'topology/flow.py', package / 'topology/core.py',
             package / 'topology/embedding.py', package / 'topology/masks.py',
             package / '_records.py']
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare(path, seeds=tuple(range(8)), levels=LEVELS):
    root = output_root(path)
    if not seeds or any(not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    levels = [float(level) for level in levels]
    if len(levels) < 2 or not np.isfinite(levels).all() or levels != sorted(set(levels)) or levels[0] != 0 or levels[-1] > .6:
        raise ValueError('Levels must be distinct increasing values from 0 through .6, including zero and a positive level')
    config = dict(schema_version=1, experiment='matched_eulerian_and_compensation',
                  seeds=[int(seed) for seed in seeds], levels=levels,
                  duration_sec=12, fps=60, samples=720, d=23, n_windows=300,
                  coefficient_field=47, density_filter=False, flow_lag_profile_power=1,
                  matching_uses_ps=False, carrier_search_hz=[.9, 1.1],
                  dependency_versions={name: version(name) for name in ['numpy', 'scipy', 'pandas', 'ripser']},
                  source_sha256=signature())
    metadata = root / 'experiment.json'
    if metadata.exists():
        if json.loads(metadata.read_text()) != config:
            raise ValueError('Experiment settings or source differ; use a new results directory')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('New experiments require an empty results directory')
        root.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(config, indent=2) + '\n')
    return root, config


def manifest(path):
    root = output_root(path)
    file = root / 'experiment.json'
    if not file.is_file():
        raise ValueError('Run the eulerian stage first in this results directory')
    config = json.loads(file.read_text())
    verified_root, verified_config = prepare(root, config['seeds'], config['levels'])
    return verified_root, verified_config


def selected_seeds(config, shard, shards):
    if not isinstance(shard, int) or not isinstance(shards, int) or shards < 1 or not 0 <= shard < shards:
        raise ValueError('Require shards >= 1 and 0 <= shard < shards')
    return config['seeds'][shard::shards]


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + '-', suffix='.csv', dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as stream:
            frame.to_csv(stream, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seed_path(root, stage, seed):
    return root / stage / ('seed_' + str(seed) + '.csv')


def read_seed(root, config, stage, seed):
    path = seed_path(root, stage, seed)
    if not path.is_file():
        raise ValueError('Missing ' + stage + ' results for seed ' + str(seed))
    frame = pd.read_csv(path, float_precision='round_trip')
    levels = config['levels'] if stage == 'eulerian' else [x for x in config['levels'] if x > 0]
    required = ['seed', 'epsilon', 'a', 'l', 'carrier_frequency', 'initial_phase', 'modulation_phase',
                'intensity', 'extent', 'frequency', 'spectral_concentration', 'ps', 'cycle_return_distance',
                'match_error_intensity', 'match_error_extent', 'match_error_frequency', 'windows_retained']
    if stage == 'compensation':
        required += ['compensation_ps_shared_period', 'compensation_ps_own_period']
    if not set(required) <= set(frame.columns):
        raise ValueError('Incomplete columns in ' + str(path))
    if len(frame) != len(levels) or frame.duplicated(['seed', 'epsilon']).any() or not frame.seed.eq(seed).all() or set(frame.epsilon) != set(levels):
        raise ValueError('Unexpected condition coverage in ' + str(path))
    if not np.isfinite(frame[required].to_numpy(float)).all() or not frame.windows_retained.eq(300).all():
        raise ValueError('Invalid measurements in ' + str(path))
    if frame[['match_error_intensity', 'match_error_extent', 'match_error_frequency']].max().max() >= 1e-6:
        raise ValueError('Matching tolerance exceeded in ' + str(path))
    return frame.sort_values('epsilon').reset_index(drop=True)


def collect(root, config, stage):
    return pd.concat([read_seed(root, config, stage, seed) for seed in config['seeds']], ignore_index=True)
