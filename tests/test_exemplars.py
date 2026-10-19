import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from prism import TopologySettings
from prism.exemplars import nearest, predict_parameters, train_decoder
from prism.exemplars.cli import main
from prism.exemplars.config import BEHAVIORS, METRICS, design, manifest, workspace
from prism.exemplars.experiment import evaluate, summarize
from prism.exemplars.model import PortableDecoder
from prism.exemplars.query import query
from prism.exemplars.validate import validate


def measured_rows(rows):
    output = []
    for source in rows:
        row = dict(source)
        f = float(row['frequency_input'])
        a = float(row['amplitude_scale'])
        r = float(row['disruption'])
        row.update(
            intensity=.02 * f * a, extent=.1 * a,
            frequency=f * (1 + .01 * r), ps=.9 - .7 * r,
            status='ok', windows_retained=300,
            image_sha256=hashlib.sha256(
                json.dumps(source, sort_keys=True).encode()
            ).hexdigest(),
        )
        output.append(row)
    return pd.DataFrame(output)


class ExemplarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.fixture_root, _ = workspace(Path(temporary.name) / 'fixture', 1)
        data = measured_rows(design())
        data.to_csv(cls.fixture_root / 'library_and_test.csv', index=False)
        train_decoder(cls.fixture_root)
        decoded = measured_rows(evaluate(data, cls.fixture_root))
        decoded.to_csv(cls.fixture_root / 'decoded.csv', index=False)
        with redirect_stdout(io.StringIO()):
            summarize(data, decoded, cls.fixture_root)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'run'
        shutil.copytree(self.fixture_root, self.root)

    def test_fixed_design(self):
        rows = pd.DataFrame(design())
        self.assertEqual(len(rows), 198)
        self.assertTrue(rows.condition_id.is_unique and rows.seed.is_unique)
        for behavior in BEHAVIORS:
            train = rows[(rows.behavior == behavior) & (rows.split == 'library')]
            test = rows[(rows.behavior == behavior) & (rows.split == 'test')]
            self.assertEqual((len(train), len(test)), (48, 18))
            for coordinate in ['frequency_input', 'amplitude_scale', 'disruption']:
                self.assertFalse(set(train[coordinate]) & set(test[coordinate]))

    def test_partial_retrieval_uses_library_behavior(self):
        result = nearest('body_rocking', [np.nan, np.nan, 1., np.nan], 3,
                         results_dir=self.root)
        self.assertTrue(result.split.eq('library').all())
        self.assertTrue(result.behavior.eq('body_rocking').all())
        data = pd.read_csv(self.root / 'library_and_test.csv')
        subset = data[data.split.eq('library') & data.behavior.eq('body_rocking')]
        self.assertAlmostEqual(abs(result.iloc[0].frequency - 1.),
                               abs(subset.frequency - 1.).min())

    def test_invalid_profiles(self):
        for target in [[np.nan] * 4, [1, 2, 3], [1, 2, np.inf, 4]]:
            with self.assertRaises(ValueError):
                nearest('body_rocking', target, results_dir=self.root)
        with self.assertRaises(ValueError):
            nearest('unknown', [1, 2, 3, 4], results_dir=self.root)
        with self.assertRaises(ValueError):
            nearest('body_rocking', [1, 2, 3, 4], 0, results_dir=self.root)
        with self.assertRaises(ValueError):
            predict_parameters('body_rocking', [np.nan, 1, 2, 3],
                               results_dir=self.root)

    def test_results_directory_is_required_by_public_apis(self):
        with self.assertRaisesRegex(TypeError, 'results_dir'):
            nearest('body_rocking', [1, 2, 3, 4])
        with self.assertRaisesRegex(TypeError, 'results_dir'):
            predict_parameters('body_rocking', [1, 2, 3, 4])
        with self.assertRaisesRegex(TypeError, 'results_dir'):
            query('body_rocking', [1, 2, 3, 4], self.root / 'query')

    def test_portable_model(self):
        data = pd.read_csv(self.root / 'library_and_test.csv')
        sample = data[data.split.eq('test')].iloc[0]
        prediction = predict_parameters(sample.behavior,
                                        sample[METRICS].to_numpy(float),
                                        results_dir=self.root)
        self.assertEqual(prediction['flow_lag_profile_power'], 1)
        self.assertEqual(set(prediction['parameters']),
                         {'frequency_input', 'amplitude_scale', 'disruption'})
        self.assertTrue(all(np.isfinite(v) for v in prediction['parameters'].values()))
        self.assertEqual(PortableDecoder.load(self.root).metadata['trainable_scalars'], 451)

    def test_training_excludes_heldout_rows(self):
        previous = PortableDecoder.load(self.root)
        path = self.root / 'library_and_test.csv'
        lines = path.read_text().splitlines(keepends=True)
        header = next(csv.reader([lines[0]]))
        edited = [lines[0]]
        for line in lines[1:]:
            values = next(csv.reader([line]))
            if values[header.index('split')] == 'test':
                for metric in METRICS:
                    values[header.index(metric)] = '1000000.0'
                stream = io.StringIO()
                csv.writer(stream, lineterminator='\n').writerow(values)
                edited.append(stream.getvalue())
            else:
                edited.append(line)
        path.write_text(''.join(edited))
        frame = pd.read_csv(path)
        metadata = train_decoder(self.root)
        current = PortableDecoder.load(self.root)
        self.assertEqual(metadata['training_rows'], 144)
        self.assertFalse(set(metadata['training_ids']) &
                         set(frame[frame.split.eq('test')].condition_id))
        for name in previous.arrays:
            np.testing.assert_allclose(previous.arrays[name], current.arrays[name],
                                       atol=1e-10, rtol=1e-10)

    def test_decoder_rejects_changed_definition_and_table(self):
        meta = manifest(self.root)
        meta['flow_lag_profile_power'] = 2
        path = self.root / 'experiment.json'
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'different measurement'):
            PortableDecoder.load(self.root)
        meta['flow_lag_profile_power'] = 1
        path.write_text(json.dumps(meta))
        with (self.root / 'library_and_test.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'CSV changed'):
            PortableDecoder.load(self.root)

    def test_workspace_default_and_mode_guard(self):
        self.assertEqual(TopologySettings().flow_lag_profile_power, 2)
        root, meta = workspace(self.root.parent / 'other')
        self.assertEqual(meta['flow_lag_profile_power'], 2)
        with self.assertRaisesRegex(ValueError, 'new results directory'):
            workspace(root, 1)

    def test_query_partial_json_and_neural_rejection(self):
        output = self.root / 'retrieval'
        result = query('body_rocking', [np.nan, np.nan, 1., np.nan], output,
                       results_dir=self.root, method='retrieval', no_video=True)
        self.assertIsNone(result['target']['intensity'])
        self.assertTrue((output / 'matches.json').exists())
        self.assertEqual(json.loads((output / 'matches.json').read_text())
                         ['flow_lag_profile_power'], 1)
        with self.assertRaisesRegex(ValueError, 'all four finite'):
            query('body_rocking', [np.nan, np.nan, 1., np.nan], self.root / 'bad',
                  results_dir=self.root, method='neural', no_video=True)
        self.assertFalse((self.root / 'bad').exists())
        with self.assertRaisesRegex(ValueError, 'different lag-distance'):
            query('body_rocking', [1, 2, 3, 4], self.root / 'mismatch',
                  results_dir=self.root, profile_power=2)

    def test_generated_run_validation_and_missing_manifest(self):
        report = validate(self.root)
        self.assertEqual(report['records'], 198)
        self.assertEqual(report['neural_outputs_remeasured'], 54)
        self.assertTrue(report['candidate_behavior_and_split_checked'])
        with self.assertRaisesRegex(ValueError, 'Run library first'):
            manifest(self.root / 'missing')

    def test_cli_requires_run_paths_and_rejects_removed_modes(self):
        commands = [
            ['init-reference'],
            ['validate'],
            ['validate', '--reference'],
            ['validate', '--results-dir', str(self.root), '--reference'],
            ['query', '--behavior', 'body_rocking', '--profile', '1', '2', '3', '4',
             '--out', str(self.root / 'query')],
        ]
        for command in commands:
            with self.subTest(command=command):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    main(command)
                self.assertEqual(caught.exception.code, 2)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['validate', '--results-dir', str(self.root)]), 0)
        self.assertEqual(json.loads(output.getvalue())['records'], 198)


if __name__ == '__main__':
    unittest.main(verbosity=2)
