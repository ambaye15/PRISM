import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from prism import TopologySettings
from prism.validation.matched_controls import run_compensation, run_eulerian, run_experiment, summarize
from prism.validation.matched_controls import eulerian
from prism.validation.matched_controls.storage import output_root, signature

FIXTURES = Path(__file__).parent / 'fixtures/matched_controls'


@pytest.fixture(scope='module')
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('matched') / 'results'
    run_experiment(root, plots=False)
    return root


def test_paper_results_match_all_archived_conditions(completed_run):
    actual = pd.read_csv(completed_run / 'eulerian.csv').sort_values(['seed', 'epsilon']).reset_index(drop=True)
    expected = pd.read_csv(FIXTURES / 'eulerian.csv').sort_values(['seed', 'epsilon']).reset_index(drop=True)
    columns = list(expected.columns)
    assert len(actual) == len(expected) == 32
    np.testing.assert_allclose(actual[columns].to_numpy(float), expected[columns].to_numpy(float), atol=1e-8, rtol=1e-8)
    assert actual.windows_retained.eq(300).all()


def test_compensation_matches_all_archived_conditions(completed_run):
    actual = pd.read_csv(completed_run / 'compensation.csv').sort_values(['seed', 'epsilon']).reset_index(drop=True)
    expected = pd.read_csv(FIXTURES / 'compensation.csv').sort_values(['seed', 'epsilon']).reset_index(drop=True)
    assert len(actual) == len(expected) == 24
    np.testing.assert_allclose(actual[list(expected.columns)].to_numpy(float), expected.to_numpy(float), atol=1e-8, rtol=1e-8)


def test_summary_preserves_the_paper_findings(completed_run):
    summary = json.loads((completed_run / 'summary.json').read_text())
    expected = json.loads((FIXTURES / 'summary.json').read_text())
    assert summary['paper_design_complete']
    assert summary['eulerian_conditions'] == expected['toy_conditions'] == 32
    assert summary['eulerian_pairs'] == expected['toy_pairs'] == 24
    assert summary['all_irregular_ps_below_matched_baseline']
    assert summary['all_seed_responses_strictly_decreasing']
    assert summary['compensation_all_pure_controls_above_irregular_ps']
    assert max(summary['max_relative_match_errors'].values()) < 1e-6
    np.testing.assert_allclose(summary['compensation_pure_ps_range'], expected['compensation_pure_ps_range'], atol=1e-8, rtol=1e-8)


def test_matching_uses_mean_distance_without_persistence(monkeypatch):
    original = eulerian._period_from_distance
    powers = []

    def period(distance, timestamps, settings):
        powers.append(settings.flow_lag_profile_power)
        return original(distance, timestamps, settings)

    def fail(*args, **kwargs):
        raise AssertionError('Matching invoked persistence')

    monkeypatch.setattr(eulerian, '_period_from_distance', period)
    monkeypatch.setattr(eulerian, 'persistence_summary', fail)
    monkeypatch.setattr(eulerian, '_window_distances', fail)
    row, _ = eulerian.matched(0, .3)
    assert powers and set(powers) == {1}
    assert max(row[key] for key in row if key.startswith('match_error_')) < 1e-6
    assert TopologySettings().flow_lag_profile_power == 2


def test_cached_shards_have_exact_coverage(completed_run):
    first = run_eulerian(completed_run, shard=0, shards=2)
    second = run_eulerian(completed_run, shard=1, shards=2)
    assert len(first) == len(second) == 16
    assert not set(first.seed) & set(second.seed)
    assert set(first.seed) | set(second.seed) == set(range(8))
    first = run_compensation(completed_run, shard=0, shards=2)
    second = run_compensation(completed_run, shard=1, shards=2)
    assert len(first) == len(second) == 12


def test_changed_protocol_and_dependency_signature_are_rejected(completed_run, monkeypatch):
    with pytest.raises(ValueError, match='new results directory'):
        run_eulerian(completed_run, seeds=[0])
    from prism.validation.matched_controls import storage
    monkeypatch.setattr(storage, 'signature', lambda: {'changed': 'source'})
    with pytest.raises(ValueError, match='new results directory'):
        summarize(completed_run, plots=False)


def test_missing_or_stale_compensation_is_reported(completed_run, tmp_path):
    root = tmp_path / 'missing'
    shutil.copytree(completed_run, root)
    path = root / 'compensation/seed_0.csv'
    path.unlink()
    with pytest.raises(ValueError, match='Missing compensation'):
        summarize(root, plots=False)
    shutil.copy2(completed_run / 'compensation/seed_0.csv', path)
    frame = pd.read_csv(path)
    frame.loc[0, 'ps'] += .01
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='do not match'):
        summarize(root, plots=False)


def test_invalid_shard_and_source_output_are_rejected(completed_run):
    for shard, shards in [(-1, 2), (2, 2), (0, 0)]:
        with pytest.raises(ValueError, match='shards'):
            run_eulerian(completed_run, shard=shard, shards=shards)
    with pytest.raises(ValueError, match='outside the prism'):
        output_root(Path(eulerian.__file__).parent / 'results')


def test_parallel_cli_and_resume_keep_source_unchanged(tmp_path):
    before = signature()
    root = tmp_path / 'parallel'
    command = [sys.executable, '-m', 'prism.validation.matched_controls', 'run',
               '--results-dir', str(root), '--seeds', '0', '1', '--levels', '0', '.6',
               '--workers', '2', '--no-plots']
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['eulerian_conditions'] == 4 and report['compensation_conditions'] == 2
    assert not report['paper_design_complete']
    stamps = {path: path.stat().st_mtime_ns for path in root.glob('*/seed_*.csv')}
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert stamps == {path: path.stat().st_mtime_ns for path in stamps}
    assert before == signature()
