import json
import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from prism import TopologySettings
from prism.exemplars import nearest, predict_parameters, train_decoder
from prism.exemplars.config import (BEHAVIORS, METRICS, REFERENCE_DIR, design,
                                    initialize_reference, manifest, workspace)
from prism.exemplars.model import PortableDecoder
from prism.exemplars.query import query
from prism.exemplars.validate import validate


class ExemplarTests(unittest.TestCase):
    def test_fixed_design(self):
        rows = pd.DataFrame(design())
        self.assertEqual(len(rows),198)
        self.assertTrue(rows.condition_id.is_unique and rows.seed.is_unique)
        for behavior in BEHAVIORS:
            train = rows[(rows.behavior == behavior) & (rows.split == "library")]
            test = rows[(rows.behavior == behavior) & (rows.split == "test")]
            self.assertEqual((len(train),len(test)),(48,18))
            for coordinate in ["frequency_input","amplitude_scale","disruption"]:
                self.assertFalse(set(train[coordinate]) & set(test[coordinate]))

    def test_partial_retrieval_uses_library_behavior(self):
        result = nearest("body_rocking",[np.nan,np.nan,1.,np.nan],3)
        self.assertTrue((result.split == "library").all())
        self.assertTrue((result.behavior == "body_rocking").all())
        data = pd.read_csv(REFERENCE_DIR/"library_and_test.csv")
        subset = data[(data.split == "library") & (data.behavior == "body_rocking")]
        self.assertAlmostEqual(abs(result.iloc[0].frequency-1.),abs(subset.frequency-1.).min())

    def test_invalid_profiles(self):
        for target in [[np.nan]*4,[1,2,3],[1,2,np.inf,4]]:
            with self.assertRaises(ValueError):
                nearest("body_rocking",target)
        with self.assertRaises(ValueError):
            nearest("unknown",[1,2,3,4])
        with self.assertRaises(ValueError):
            nearest("body_rocking",[1,2,3,4],0)
        with self.assertRaises(ValueError):
            predict_parameters("body_rocking",[np.nan,1,2,3])

    def test_portable_model(self):
        data = pd.read_csv(REFERENCE_DIR/"library_and_test.csv")
        sample = data[data.split == "test"].iloc[0]
        prediction = predict_parameters(sample.behavior,sample[METRICS].to_numpy(float))
        self.assertEqual(prediction["flow_lag_profile_power"],1)
        self.assertEqual(set(prediction["parameters"]),{"frequency_input","amplitude_scale","disruption"})
        self.assertTrue(all(np.isfinite(v) for v in prediction["parameters"].values()))
        self.assertEqual(PortableDecoder.load(REFERENCE_DIR).metadata["trainable_scalars"],451)

    def test_training_excludes_heldout_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = initialize_reference(Path(temp)/"run")
            previous = PortableDecoder.load(root)


            lines = (root/"library_and_test.csv").read_text().splitlines(keepends=True)
            header = next(csv.reader([lines[0]]))
            edited = [lines[0]]
            for line in lines[1:]:
                values = next(csv.reader([line]))
                if values[header.index("split")] == "test":
                    for metric in METRICS:
                        values[header.index(metric)] = "1000000.0"
                    stream = io.StringIO()
                    csv.writer(stream,lineterminator="\n").writerow(values)
                    edited.append(stream.getvalue())
                else:
                    edited.append(line)
            (root/"library_and_test.csv").write_text("".join(edited))
            frame = pd.read_csv(root/"library_and_test.csv")
            metadata = train_decoder(root)
            current = PortableDecoder.load(root)
            self.assertEqual(metadata["training_rows"],144)
            self.assertFalse(set(metadata["training_ids"]) & set(frame[frame.split == "test"].condition_id))
            for name in previous.arrays:
                np.testing.assert_allclose(previous.arrays[name],current.arrays[name],atol=1e-10,rtol=1e-10)

    def test_decoder_rejects_changed_definition_and_table(self):
        with tempfile.TemporaryDirectory() as temp:
            root = initialize_reference(Path(temp)/"run")
            meta = manifest(root)
            meta["flow_lag_profile_power"] = 2
            (root/"experiment.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,"different measurement"):
                PortableDecoder.load(root)
            meta["flow_lag_profile_power"] = 1
            (root/"experiment.json").write_text(json.dumps(meta))
            with (root/"library_and_test.csv").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError,"CSV changed"):
                PortableDecoder.load(root)

    def test_workspace_default_and_mode_guard(self):
        self.assertEqual(TopologySettings().flow_lag_profile_power,2)
        with tempfile.TemporaryDirectory() as temp:
            root, meta = workspace(Path(temp)/"run")
            self.assertEqual(meta["flow_lag_profile_power"],2)
            with self.assertRaisesRegex(ValueError,"new results directory"):
                workspace(root,1)

    def test_query_partial_json_and_neural_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"retrieval"
            result = query("body_rocking",[np.nan,np.nan,1.,np.nan],output,
                           method="retrieval",no_video=True)
            self.assertIsNone(result["target"]["intensity"])
            self.assertTrue((output/"matches.json").exists())
            self.assertEqual(json.loads((output/"matches.json").read_text())["flow_lag_profile_power"],1)
            with self.assertRaisesRegex(ValueError,"all four finite"):
                query("body_rocking",[np.nan,np.nan,1.,np.nan],Path(temp)/"bad",
                      method="neural",no_video=True)
            self.assertFalse((Path(temp)/"bad").exists())
            with self.assertRaisesRegex(ValueError,"different lag-distance"):
                query("body_rocking",[1,2,3,4],Path(temp)/"mismatch",profile_power=2)

    def test_reference_validation_and_no_overwrite(self):
        report = validate(REFERENCE_DIR)
        self.assertEqual(report["records"],198)
        self.assertEqual(report["neural_outputs_remeasured"],54)
        with tempfile.TemporaryDirectory() as temp:
            root = initialize_reference(Path(temp)/"run")
            with self.assertRaisesRegex(ValueError,"empty or new"):
                initialize_reference(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
