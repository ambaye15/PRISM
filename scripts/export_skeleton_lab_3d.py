import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_FILES = (
    "joints_3d.npy",
    "keypoints_3d.npy",
    "world_joints.npy",
)
ARRAY_KEYS = ("joints_3d", "keypoints_3d", "world_joints", "joints")


def find_array(bundle):
    for name in ARRAY_FILES:
        path = bundle / name
        if path.exists():
            return np.load(path)
    for path in bundle.glob("*.npz"):
        data = np.load(path, allow_pickle=True)
        for key in ARRAY_KEYS:
            if key in data.files:
                return np.asarray(data[key])
    raise FileNotFoundError(
        f"{bundle}: no exported 3D joint array. Save the Skeleton Lab world "
        "joint trajectory as joints_3d.npy or an NPZ key named joints_3d."
    )


def simulate_array(bundle):

    spec_path = bundle / "spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(str(bundle) + ": spec.json is unavailable")
    try:
        from skeleton_lab.simulate import simulate
        from skeleton_lab.spec import ScenarioSpec
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Skeleton Lab is required to regenerate old bundles. Install the "
            "PRISM validation extra."
        ) from error
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    result = simulate(ScenarioSpec.model_validate(payload))
    names = list(result.joints_world)
    return np.stack([result.joints_world[name] for name in names], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    bundles = sorted(path.parent for path in args.bundles_root.rglob("truth.json"))
    if not bundles:
        raise FileNotFoundError("no Skeleton Lab truth.json bundles found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped = []
    for bundle in bundles:
        try:
            joints = find_array(bundle)
        except FileNotFoundError as error:
            try:
                joints = simulate_array(bundle)
            except (FileNotFoundError, RuntimeError, ValueError) as fallback_error:
                skipped.append(str(error) + "; fallback: " + str(fallback_error))
                continue
        truth = json.loads((bundle / "truth.json").read_text())
        np.savez_compressed(
            args.output_root / f"{bundle.name}.npz",
            joints_3d=joints,
            fps=float(truth["fps"]),
            condition_id=bundle.name,
            behavior=truth.get("behavior", "unknown"),
        )
        exported += 1
    print(f"Exported {exported} trajectories; skipped {len(skipped)} bundles")
    if skipped:
        print(skipped[0])


if __name__ == "__main__":
    main()
