import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BEHAVIORS, METRICS, PARAMETERS, LOWER, UPPER,
                     csv_hash, manifest, scales_for)


def inputs(frame):
    return np.c_[frame[METRICS].to_numpy(float),
                 np.asarray([[float(b == x) for x in BEHAVIORS] for b in frame.behavior])]


class PortableDecoder:

    def __init__(self, arrays, metadata):
        self.arrays, self.metadata = arrays, metadata
        expected = {"xmean":(7,),"xscale":(7,),"ymean":(3,),"yscale":(3,),
                    "w0":(7,16),"b0":(16,),"w1":(16,16),"b1":(16,),"w2":(16,3),"b2":(3,)}
        for name, shape in expected.items():
            if name not in arrays or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"Invalid decoder array {name}")
        if (arrays["xscale"] <= 0).any() or (arrays["yscale"] <= 0).any():
            raise ValueError("Decoder scales must be positive")
        if metadata.get("behaviors") != BEHAVIORS or metadata.get("metrics") != METRICS:
            raise ValueError("Decoder input coordinate order is incompatible")
        if metadata.get("parameters") != PARAMETERS:
            raise ValueError("Decoder parameter order is incompatible")

    @classmethod
    def load(cls, results_dir):
        root = Path(results_dir)
        with np.load(root/"decoder.npz", allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files if k != "metadata"}
            meta = json.loads(z["metadata"].item())
        if meta["library_csv_sha256"] != csv_hash(root):
            raise ValueError("Library CSV changed after training; retrain the decoder")
        if meta["flow_lag_profile_power"] != manifest(root)["flow_lag_profile_power"]:
            raise ValueError("Decoder and experiment use different measurement definitions")
        return cls(arrays, meta)

    def predict(self, x):
        a = self.arrays
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
            raise ValueError("Decoder input must be finite [N,7]")
        h = (x-a["xmean"])/a["xscale"]
        for i in range(2):
            h = np.tanh(h @ a[f"w{i}"] + a[f"b{i}"])
        return (h @ a["w2"] + a["b2"]) * a["yscale"] + a["ymean"]


def train_decoder(results_dir):

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from importlib.metadata import version
    from threadpoolctl import threadpool_limits
    root = Path(results_dir)
    meta = manifest(root)
    data = pd.read_csv(root/"library_and_test.csv")
    library = data[(data.split == "library") & (data.status == "ok")].copy()
    if set(library.behavior) != set(BEHAVIORS):
        raise ValueError("Training requires usable library rows for all three supported behaviors")
    if not np.isfinite(library[METRICS+PARAMETERS].to_numpy(float)).all():
        raise ValueError("Training records contain unavailable coordinates")
    if not library.condition_id.is_unique:
        raise ValueError("Training condition IDs must be unique")
    xscale = StandardScaler().fit(inputs(library))
    yscale = StandardScaler().fit(library[PARAMETERS])
    model = MLPRegressor(hidden_layer_sizes=(16,16), activation="tanh", solver="lbfgs",
                         alpha=.01, max_iter=3000, random_state=20261017)
    with threadpool_limits(limits=1):
        model.fit(xscale.transform(inputs(library)), yscale.transform(library[PARAMETERS]))
    arrays = {"xmean":xscale.mean_,"xscale":xscale.scale_,
              "ymean":yscale.mean_,"yscale":yscale.scale_}
    for i,(w,b) in enumerate(zip(model.coefs_, model.intercepts_)):
        arrays[f"w{i}"], arrays[f"b{i}"] = w,b
    metadata = {"schema_version":1,"behaviors":BEHAVIORS,"metrics":METRICS,"parameters":PARAMETERS,
                "library_csv_sha256":csv_hash(root), "flow_lag_profile_power":meta["flow_lag_profile_power"],
                "training_ids":library.condition_id.tolist(),"training_rows":len(library),
                "hidden_layers":[16,16],"activation":"tanh","solver":"lbfgs","alpha":.01,
                "max_iter":3000,"random_state":20261017,"n_iter":model.n_iter_,
                "trainable_scalars":sum(w.size for w in model.coefs_+model.intercepts_),
                "sklearn_version":version("scikit-learn"),
                "interpretation":"Illustrative simulator settings; not participant reconstruction"}
    decoder = PortableDecoder(arrays, metadata)
    check = decoder.predict(inputs(library))
    expected = yscale.inverse_transform(model.predict(xscale.transform(inputs(library))))
    np.testing.assert_allclose(check, expected, atol=1e-12, rtol=1e-12)
    metadata["portable_vs_sklearn_max_abs_error"] = float(np.max(np.abs(check-expected)))
    np.savez_compressed(root/"decoder.npz", **arrays, metadata=json.dumps(metadata))
    scales = scales_for(library)
    (root/"scales.json").write_text(json.dumps({k:v.tolist() for k,v in scales.items()},indent=2))
    (root/"decoder_training.json").write_text(json.dumps(metadata,indent=2))
    return metadata


def predict_parameters(behavior, target, *, results_dir):
    if behavior not in BEHAVIORS:
        raise ValueError(f"Unsupported behavior: {behavior}")
    target = np.asarray(target, float)
    if target.shape != (4,) or not np.isfinite(target).all():
        raise ValueError("The neural decoder requires four finite measurements")
    root = Path(results_dir)
    decoder = PortableDecoder.load(root)
    x = np.r_[target, [float(behavior == b) for b in BEHAVIORS]][None]
    raw = decoder.predict(x)[0]
    clipped = np.clip(raw, LOWER, UPPER)
    return {"parameters":dict(zip(PARAMETERS, clipped.tolist())),
            "raw_parameters":dict(zip(PARAMETERS, raw.tolist())),
            "parameter_clipped":bool(np.any(raw != clipped)),
            "flow_lag_profile_power":decoder.metadata["flow_lag_profile_power"]}
