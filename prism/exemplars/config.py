import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
SPEC_DIR = ASSET_DIR / "specs"
REFERENCE_DIR = ASSET_DIR / "reference"
BEHAVIORS = ["body_rocking", "arm_flapping", "hand_flapping"]
DRIVES = {"body_rocking": "rock", "arm_flapping": "flap", "hand_flapping": "flap"}
METRICS = ["intensity", "extent", "frequency", "ps"]
PARAMETERS = ["frequency_input", "amplitude_scale", "disruption"]
FPS, DURATION, WARMUP, N_WINDOWS = 20., 8., 2., 300
LOWER = np.array([.75, .6, 0.])
UPPER = np.array([2., 1.4, .75])


def design():
    rows = []
    for behavior in BEHAVIORS:
        for split, frequencies, amplitudes, disruptions in [
            ("library", [.75, 1., 1.5, 2.], [.6, 1., 1.4], [0., .25, .5, .75]),
            ("test", [.875, 1.25, 1.75], [.8, 1.2], [.125, .375, .625])]:
            for j, (f, a, r) in enumerate(itertools.product(frequencies, amplitudes, disruptions)):
                seed = (1000 if split == "library" else 2000) + BEHAVIORS.index(behavior)*100 + j
                rows.append(dict(condition_id=f"{behavior}__{split}__{j:03d}", behavior=behavior,
                    split=split, seed=seed, frequency_input=f, amplitude_scale=a, disruption=r))
    return rows


def scales_for(library):
    scales = {}
    for behavior, group in library.groupby("behavior"):
        v = group[METRICS].to_numpy(float)
        q25, q75 = np.quantile(v, [.25, .75], axis=0)
        scales[behavior] = np.maximum(q75-q25, [1e-3, 1e-3, .05, .05])
    return scales


def csv_hash(results_dir):
    return hashlib.sha256((Path(results_dir)/"library_and_test.csv").read_bytes()).hexdigest()


def signature(power):
    from importlib.metadata import version
    root = Path(__file__).resolve().parents[1]
    paths = [root/"optical_flow.py", root/"topology/flow.py", root/"topology/masks.py",
             root/"topology/embedding.py", root/"topology/period.py", root/"topology/core.py",
             Path(__file__).with_name("simulation.py")]
    text = "".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    text += "".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SPEC_DIR.glob("*.json")))
    text += json.dumps({"lag_profile_power":power,"windows":N_WINDOWS,"fps":FPS,"duration":DURATION})
    for dependency in ["numpy", "scipy", "opencv-python-headless", "ripser", "skeleton-lab"]:
        text += dependency + version(dependency)
    return hashlib.sha256(text.encode()).hexdigest()


def manifest(results_dir):
    path = Path(results_dir)/"experiment.json"
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run init-reference or library first.")
    data = json.loads(path.read_text())
    if data.get("flow_lag_profile_power") not in (1, 2):
        raise ValueError("Experiment must explicitly declare lag profile power 1 or 2")
    return data


def workspace(results_dir, profile_power=None):
    out = Path(results_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"experiment.json").exists():
        data = manifest(out)
        if profile_power is not None and data["flow_lag_profile_power"] != profile_power:
            raise ValueError("Measurement definition differs: use a new results directory")
    else:
        power = 2 if profile_power is None else profile_power
        if power not in (1, 2):
            raise ValueError("profile_power must be 1 or 2")
        data = {"schema_version":1, "flow_lag_profile_power":power,
                "source":"new experiment using installed PRISM", "archived_reference":False}
        (out/"experiment.json").write_text(json.dumps(data, indent=2))
    return out, data


def initialize_reference(results_dir):
    import shutil
    out = Path(results_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("init-reference requires an empty or new results directory")
    shutil.copytree(REFERENCE_DIR, out, dirs_exist_ok=True)
    return out
