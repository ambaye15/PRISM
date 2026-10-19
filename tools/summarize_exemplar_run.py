import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('runs/reference'))
    parser.add_argument('--output-root', type=Path, default=Path('paper_updates/exemplars'))
    args = parser.parse_args()
    root = args.results_dir
    out = args.output_root
    metrics = ['intensity', 'extent', 'frequency', 'ps']
    data = pd.read_csv(root / 'library_and_test.csv').set_index('condition_id')
    decoded = pd.read_csv(root / 'decoded.csv')
    scores = pd.read_csv(root / 'evaluation.csv')
    scales = json.loads((root / 'scales.json').read_text())
    training = json.loads((root / 'decoder_training.json').read_text())
    assert len(data) == 198 and len(decoded) == 54 and len(scores) == 216
    assert data.status.eq('ok').all() and decoded.status.eq('ok').all()
    assert np.isfinite(data[metrics]).all().all()
    assert np.isfinite(decoded[metrics]).all().all()
    assert decoded.parameter_clipped.astype(str).str.lower().isin(['true', 'false']).all()
    for row in decoded.to_dict('records'):
        delta = abs(np.array([row[k] for k in metrics]) - data.loc[row['query_id'], metrics].to_numpy(float))
        expected = float(np.sqrt(np.mean((delta / np.array(scales[row['behavior']])) ** 2)))
        match = scores[(scores.query_id == row['query_id']) & (scores.method == 'neural_decoder')]
        assert len(match) == 1
        assert np.isclose(match.profile_error.iloc[0], expected, atol=1e-12, rtol=1e-12)
        assert np.allclose(match[[k + '_abs_error' for k in metrics]].iloc[0].to_numpy(float), delta, atol=1e-12, rtol=1e-12)
    out.mkdir(parents=True, exist_ok=True)
    grouped = scores.groupby('method')
    summary = grouped.agg(n=('profile_error', 'size'), profile_error=('profile_error', 'median'),
                          intensity_mae=('intensity_abs_error', 'mean'), extent_mae=('extent_abs_error', 'mean'),
                          frequency_mae=('frequency_abs_error', 'mean'), ps_mae=('ps_abs_error', 'mean'))
    extended = grouped.agg(n=('profile_error', 'size'), median_profile_error=('profile_error', 'median'),
                           q90_profile_error=('profile_error', lambda x: x.quantile(.9)),
                           max_profile_error=('profile_error', 'max'),
                           within_035=('profile_error', lambda x: int((x <= .35).sum())),
                           frequency_mae=('frequency_abs_error', 'mean'), ps_mae=('ps_abs_error', 'mean'))
    geometry = scores.groupby(['behavior', 'method']).agg(
        n=('profile_error', 'size'), median_profile_error=('profile_error', 'median'),
        frequency_mae=('frequency_abs_error', 'mean'), ps_mae=('ps_abs_error', 'mean'))
    summary.to_csv(out / 'summary.csv')
    extended.to_csv(out / 'extended_summary.csv')
    geometry.to_csv(out / 'geometry_summary.csv')
    stats = dict(trainable_scalars=training['trainable_scalars'], optimizer_iterations=training['n_iter'],
                 training_loss=None, model_bytes=(root / 'decoder.npz').stat().st_size,
                 inference_median_ms=None, inference_q95_ms=None,
                 generated_available=int(decoded.status.eq('ok').sum()),
                 parameter_clipped=int(decoded.parameter_clipped.astype(str).str.lower().eq('true').sum()),
                 note='Optimizer loss and inference timing were not recorded for this checkpoint; older timing and loss were not carried forward.')
    (out / 'decoder_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    source = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
              ['decoder.npz', 'decoder_training.json', 'library_and_test.csv', 'decoded.csv', 'evaluation.csv', 'scales.json']}
    (out / 'source_sha256.json').write_text(json.dumps(source, indent=2) + '\n')
    illustrated = []
    indexed = decoded.set_index('query_id')
    for query in ['body_rocking__test__014', 'body_rocking__test__003']:
        illustrated.append(dict(query_id=query,
                                target={k: float(data.loc[query, k]) for k in metrics},
                                achieved={k: float(indexed.loc[query, k]) for k in metrics}))
    (out / 'illustrated_examples.json').write_text(json.dumps(illustrated, indent=2) + '\n')
    print(summary.to_string())
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
