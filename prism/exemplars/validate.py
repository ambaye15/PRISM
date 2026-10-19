import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import design, BEHAVIORS, METRICS, N_WINDOWS, manifest
from .model import PortableDecoder
from .query import nearest


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def validate(results_dir):
    root = Path(results_dir)
    meta = manifest(root)
    rows = pd.DataFrame(design())
    require(len(rows) == 198 and rows.condition_id.is_unique and rows.seed.is_unique, "Invalid design")
    for behavior in BEHAVIORS:
        train = rows[(rows.behavior == behavior) & (rows.split == "library")]
        test = rows[(rows.behavior == behavior) & (rows.split == "test")]
        require(len(train) == 48 and len(test) == 18, "Incorrect split sizes")
        for k in ["frequency_input","amplitude_scale","disruption"]:
            require(not set(train[k]) & set(test[k]), f"Generating axis overlaps: {k}")
    report = {"design_counts_checked":True,"all_three_parameter_axes_held_out":True,
              "flow_lag_profile_power":meta["flow_lag_profile_power"],
              "archived_reference":meta.get("archived_reference", False)}
    if not (root/"library_and_test.csv").exists():
        return report
    data = pd.read_csv(root/"library_and_test.csv")
    require(len(data) == 198 and data.condition_id.is_unique, "Invalid condition table")
    train, test = data[data.split == "library"], data[data.split == "test"]
    require(not set(train.image_sha256.dropna()) & set(test.image_sha256.dropna()), "Image leakage")
    complete = data[data.status == "ok"]
    require(np.isfinite(complete[METRICS]).all().all(), "Non-finite usable profile")
    require((complete.windows_retained == N_WINDOWS).all(), "Unexpected window support")
    report.update(records=len(data), available=len(complete), duplicate_images_across_splits=0)
    if (root/"decoder.npz").exists():
        decoder = PortableDecoder.load(root)
        require(set(decoder.metadata["training_ids"]) <= set(train.condition_id), "Decoder trained on held-out IDs")
        require(not set(decoder.metadata["training_ids"]) & set(test.condition_id), "Decoder training leakage")
        report["decoder_training_rows"] = decoder.metadata["training_rows"]
    if (root/"evaluation.csv").exists():
        scores = pd.read_csv(root/"evaluation.csv")
        require(not scores.duplicated(["query_id","method"]).any(), "Duplicate evaluations")
        index = data.set_index("condition_id")
        for r in scores[scores.method == "retrieval"].to_dict("records"):
            require(index.loc[r["candidate_id"],"split"] == "library", "Retrieved a held-out candidate")
            require(index.loc[r["candidate_id"],"behavior"] == index.loc[r["query_id"],"behavior"], "Behavior mismatch")
        decoded = pd.read_csv(root/"decoded.csv")
        require(set(decoded.query_id) == set(test.condition_id), "Missing decoded conditions")
        require((decoded[decoded.status == "ok"].windows_retained == N_WINDOWS).all(), "Decoded window mismatch")
        require(not set(decoded.seed) & set(data.seed), "Decoded seeds overlap")
        require(not set(decoded.image_sha256.dropna()) & set(data.image_sha256.dropna()), "Decoded image overlap")
        require(set(scores[scores.method == "neural_decoder"].query_id) ==
                set(decoded[decoded.status == "ok"].query_id), "Decoded availability mismatch")
        report.update(evaluation_rows=len(scores), neural_outputs_remeasured=int((decoded.status == "ok").sum()),
                      candidate_behavior_and_split_checked=True)
    if (root/"scales.json").exists():
        item = nearest("body_rocking", [np.nan,np.nan,1.,np.nan], 1, results_dir=root).iloc[0]
        usable = train[(train.behavior == "body_rocking") & (train.status == "ok")]
        require(np.isclose(abs(item.frequency-1.), abs(usable.frequency-1.).min()), "Missing-coordinate matching failed")
        try:
            nearest("body_rocking", np.full(4,np.nan), results_dir=root)
        except ValueError:
            pass
        else:
            raise ValueError("All-missing profiles were accepted")
        report["missing_coordinate_matching_checked"] = True
    return report
