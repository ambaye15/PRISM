import json

import numpy as np

from .eulerian import LEVELS, NU, T
from .storage import collect, manifest, write_csv


def cycle_duration_cv(row):
    phase = 2 * np.pi * row['carrier_frequency'] * T + row['initial_phase'] + row['epsilon'] * np.sin(2 * np.pi * NU * T + row['modulation_phase'])
    levels = 2 * np.pi * np.arange(np.ceil(phase[0] / (2 * np.pi)), np.floor(phase[-1] / (2 * np.pi)) + 1)
    periods = np.diff(np.interp(levels, phase, T))
    return float(np.std(periods) / np.mean(periods))


def summarize(results_dir, plots=True):
    root, config = manifest(results_dir)
    primary = collect(root, config, 'eulerian')
    compensation = collect(root, config, 'compensation')
    original = primary[primary.epsilon > 0].sort_values(['seed', 'epsilon']).reset_index(drop=True)
    aligned = compensation.sort_values(['seed', 'epsilon']).reset_index(drop=True)
    shared = [column for column in primary if column in compensation]
    if not np.allclose(original[shared].to_numpy(float), aligned[shared].to_numpy(float), atol=1e-12, rtol=1e-12):
        raise ValueError('Compensation results do not match the current Eulerian rows; rerun compensation with --force')
    primary['cycle_duration_cv'] = primary.apply(cycle_duration_cv, axis=1)
    cycles = primary.set_index(['seed', 'epsilon']).cycle_duration_cv
    compensation['cycle_duration_cv'] = compensation.apply(lambda row: cycles.loc[(row.seed, row.epsilon)], axis=1)
    baseline = primary[primary.epsilon == 0].set_index('seed')
    primary['ps_drop'] = primary.apply(lambda row: baseline.loc[row.seed, 'ps'] - row.ps, axis=1)
    summary = primary.groupby('epsilon').agg(
        n=('seed', 'size'), intensity=('intensity', 'mean'), extent=('extent', 'mean'),
        frequency=('frequency', 'mean'), cycle_duration_cv=('cycle_duration_cv', 'mean'),
        ps_mean=('ps', 'mean'), ps_min=('ps', 'min'), ps_max=('ps', 'max'),
        mean_ps_drop=('ps_drop', 'mean'), cycle_return_distance=('cycle_return_distance', 'mean'),
        spectral_concentration=('spectral_concentration', 'mean')).reset_index()
    irregular = primary[primary.epsilon > 0]
    errors = primary[['match_error_intensity', 'match_error_extent', 'match_error_frequency']].max().to_dict()
    info = dict(eulerian_conditions=len(primary), eulerian_pairs=len(irregular),
                compensation_conditions=len(compensation), flow_lag_profile_power=1,
                paper_design_complete=config['seeds'] == list(range(8)) and config['levels'] == list(LEVELS),
                max_relative_match_errors={key: float(value) for key, value in errors.items()},
                all_irregular_ps_below_matched_baseline=bool(irregular.ps_drop.gt(0).all()),
                all_seed_responses_strictly_decreasing=bool(all(np.all(np.diff(group.sort_values('epsilon').ps) < 0) for _, group in primary.groupby('seed'))),
                compensation_all_pure_controls_above_irregular_ps=bool((compensation.compensation_ps_shared_period > compensation.ps).all()),
                compensation_pure_ps_range=[float(compensation.compensation_ps_shared_period.min()), float(compensation.compensation_ps_shared_period.max())])
    write_csv(primary, root / 'eulerian.csv')
    write_csv(compensation, root / 'compensation.csv')
    write_csv(summary, root / 'by_level.csv')
    (root / 'summary.json').write_text(json.dumps(info, indent=2) + '\n')
    text = ['# Matched Eulerian controls and compensation', '',
            'This run matches finite-sample intensity, silhouette extent and estimated frequency before computing persistence. PS is not used for matching.', '',
            summary.to_string(index=False), '',
            'The exact Eulerian field includes moving-support overlap. It does not reproduce image estimation, validity weighting or clipping.', '',
            'The phase seeds are controlled synthetic examples, not independent human participants. Cycle-return distance and spectral concentration are comparison descriptors; these results do not establish unique optimality of PS.', '',
            'The compensation diagnostic retains adjusted amplitude, moving-region height and carrier frequency while removing phase modulation. It was added after the primary experiment to assess the effect of geometric compensation.', '',
            json.dumps(info, indent=2), '']
    (root / 'RESULTS.md').write_text('\n'.join(text))
    if plots:
        from prism.plotting.matched_controls import plot_results
        plot_results(primary, compensation, root)
    return info
