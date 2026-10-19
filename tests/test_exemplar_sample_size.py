import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.exemplars import experiment
from prism.exemplars import sample_size as study
from prism.exemplars.config import BEHAVIORS, METRICS, PARAMETERS, design


def measured(rows, *args, **kwargs):
    output = []
    for source in rows:
        row = dict(source)
        f, a, r = [float(row[key]) for key in PARAMETERS]
        row.update(status='ok', intensity=.02 * f * a, extent=.1 * a,
                   frequency=f * (1 + .01 * r), ps=.9 - .7 * r,
                   windows_retained=300,
                   image_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest())
        output.append(row)
    return pd.DataFrame(output).sort_values('condition_id')


def test_nested_design_keeps_original_targets_and_original_library():
    previous = None
    original = {row['condition_id']: row for row in design()}
    for size in [144, 576, 1440]:
        rows = study.expanded_design(size)
        lookup = {row['condition_id']: row for row in rows}
        assert len(rows) == size + 54
        assert len({row['seed'] for row in rows}) == len(rows)
        assert all(lookup[key] == value for key, value in original.items())
        assert all(sum(row['split'] == 'library' and row['behavior'] == behavior
                       for row in rows) == size // 3 for behavior in BEHAVIORS)
        assert study.check_data(measured(rows), rows)['training_samples'] == size
        if previous is not None:
            assert all(lookup[key] == value for key, value in previous.items())
        previous = lookup


@pytest.mark.parametrize('size', [0, 143, 145, 144.0])
def test_invalid_training_size_is_rejected(size):
    with pytest.raises(ValueError, match='multiples of three'):
        study.expanded_design(size)


def test_sampler_is_reproducible_and_seed_changes_only_new_examples():
    a = study.expanded_design(576, 10)
    b = study.expanded_design(576, 10)
    c = study.expanded_design(576, 11)
    assert a == b
    assert a[:198] == c[:198] == design()
    assert a[198:] != c[198:]


def test_coordinate_and_image_leakage_are_rejected():
    rows = study.expanded_design(147)
    rows[-1]['frequency_input'] = 1.25
    with pytest.raises(ValueError, match='Held-out axis overlaps'):
        study.check_data(measured(rows), rows)
    rows = study.expanded_design(147)
    data = measured(rows)
    first_train = data.index[data.split.eq('library')][0]
    first_test = data.index[data.split.eq('test')][0]
    data.loc[first_train, 'image_sha256'] = data.loc[first_test, 'image_sha256']
    with pytest.raises(ValueError, match='image overlap'):
        study.check_data(data, rows)


def test_reuse_accepts_only_matching_pipeline_and_design(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    (source / 'records').mkdir(parents=True)
    target.mkdir()
    rows = design()[:3]
    for index, row in enumerate(rows):
        record = dict(status='ok', pipeline_signature='current' if index != 1 else 'old',
                      input_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
        if index == 2:
            record['input_row_sha256'] = 'changed-parameters'
        (source / 'records' / (row['condition_id'] + '.json')).write_text(json.dumps(record))
    assert study.reuse_records(source, target, rows, 'current') == 1
    assert len(list((target / 'records').glob('*.json'))) == 1
    assert study.reuse_records(source, target, rows, 'current') == 0


def test_retrieval_ranking_uses_supplied_fixed_scales():
    data = pd.DataFrame([
        dict(condition_id='A', behavior='body_rocking', split='library', intensity=10., extent=0., frequency=0., ps=0.),
        dict(condition_id='B', behavior='body_rocking', split='library', intensity=0., extent=2., frequency=0., ps=0.),
        dict(condition_id='Q', behavior='body_rocking', split='test', intensity=0., extent=0., frequency=0., ps=0.),
    ])
    first = study.fixed_retrieval(data, {'body_rocking': [100., 1., 1., 1.]})
    second = study.fixed_retrieval(data, {'body_rocking': [1., 100., 1., 1.]})
    selected = first[first.method.eq('retrieval')].iloc[0]
    assert selected.candidate_id == 'A'
    assert selected.profile_error == pytest.approx(.05)
    assert second[second.method.eq('retrieval')].iloc[0].candidate_id == 'B'


@pytest.fixture(scope='module')
def finished_study(tmp_path_factory):
    root = tmp_path_factory.mktemp('sample_size') / 'results'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, 'signature', lambda power: 'test-measurement-pipeline')
        patch.setattr(study, 'batch', measured)
        patch.setattr(experiment, 'batch', measured)
        study.run_study(root, training_sizes=[144, 147], workers=1, make_plots=False)
        yield root


def test_full_orchestration_fits_real_fixed_architecture_and_remeasures(finished_study):
    root = finished_study
    reports = json.loads((root / 'validation.json').read_text())
    assert set(reports) == {'144', '147'}
    for key, report in reports.items():
        assert report['training_rows'] == int(key)
        assert report['test_samples'] == report['neural_outputs_remeasured'] == 54
        assert report['trainable_scalars'] == 451
        assert report['profile_scales_fixed']
    scales = json.loads((root / 'reference_scales.json').read_text())
    assert json.loads((root / 'train_144/scales.json').read_text()) == scales
    assert json.loads((root / 'train_147/scales.json').read_text()) == scales
    frame = pd.read_csv(root / 'summary.csv')
    assert len(frame) == 8 and frame.n.eq(54).all()
    assert np.isfinite(frame[['median_profile_error', 'frequency_mae', 'ps_mae']]).all().all()


def test_resume_keeps_models_and_rejects_protocol_changes(finished_study):
    root = finished_study
    before = {size: (root / f'train_{size}/decoder.npz').read_bytes() for size in [144, 147]}
    study.run_study(root, training_sizes=[147, 144], workers=1, make_plots=False)
    assert all((root / f'train_{size}/decoder.npz').read_bytes() == data
               for size, data in before.items())
    with pytest.raises(ValueError, match='settings or code changed'):
        study.run_study(root, training_sizes=[144, 147], workers=1, sampler_seed=99, make_plots=False)


def test_validator_rejects_changed_neural_scores(finished_study):
    root = finished_study / 'train_147'
    path = root / 'evaluation.csv'
    saved = path.read_bytes()
    data = pd.read_csv(path)
    data.loc[data.method.eq('neural_decoder'), 'profile_error'] += .1
    data.to_csv(path, index=False)
    scales = json.loads((finished_study / 'reference_scales.json').read_text())
    try:
        with pytest.raises(ValueError, match='not correctly remeasured'):
            study.validate_model(root, 147, 20261019, scales)
    finally:
        path.write_bytes(saved)


def test_plots_can_be_written_from_completed_study(finished_study):
    from prism.plotting.exemplar_sample_size import plot_sample_size
    plot_sample_size(pd.read_csv(finished_study / 'summary.csv'), finished_study)
    assert (finished_study / 'sample_size.pdf').stat().st_size > 1000
    assert (finished_study / 'sample_size.png').stat().st_size > 1000
