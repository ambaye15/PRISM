"""Bundle-level and sweep-level analysis."""


import json
from pathlib import Path

import numpy as np
import pandas as pd

from .masks import render_exact_sequence
from .measurements import optical_flow_intensity, production_flow_output, topology_measurements


def extract_truth(bundle , drive  = "flap", extent_dof  = "shoulder_l.abduction")   :
    truth = json.loads((bundle / "truth.json").read_text(encoding="utf-8"))
    drive_truth = truth["drives"][drive]
    speeds = np.asarray(list(truth["keypoints"]["per_joint_rms_speed_px_s"].values()), dtype=float)
    body_height = float(truth["keypoints"]["body_height_px"])
    frequency = float(drive_truth["mean_frequency_hz"])
    amplitude = float(drive_truth["mean_amplitude"])
    return {
        "bundle": bundle.name,
        "bundle_path": str(bundle),
        "seed": int(truth["seed"]),
        "fps": float(truth["fps"]),
        "frequency_truth_hz": frequency,
        "amplitude_truth": amplitude,
        "amplitude_x_frequency": amplitude * frequency,
        "intensity_truth_height_s": float(np.quantile(speeds, 0.75) / body_height),
        "extent_truth_deg": float(truth["dofs"].get(extent_dof, {}).get("peak_to_peak", np.nan)),
        "camera_azimuth_deg": float(truth["camera"]["azimuth_deg"]),
        "camera_elevation_deg": float(truth["camera"]["elevation_deg"]),
        "camera_shake_sigma_px": float(truth["camera"]["shake_sigma_px"]),
    }


def analyze_bundle(
    bundle ,
    drive  = "flap",
    extent_dof  = "shoulder_l.abduction",
    dimension  = 23,
    n_points  = 300,
    flow_variant  = "skeleton",
    mask_variant  = "simple",
    max_dimension  = 256,
)   :
    bundle = Path(bundle)
    frames, masks, fps = render_exact_sequence(
        bundle, flow_variant=flow_variant, mask_variant=mask_variant,
        max_dimension=max_dimension
    )
    row = extract_truth(bundle, drive=drive, extent_dof=extent_dof)
    flow_output = production_flow_output(frames, masks, fps)
    row.update(optical_flow_intensity(frames, masks, fps, flow_output=flow_output))
    row.update(topology_measurements(
        frames, masks, fps, dimension=dimension, n_points=n_points,
        flow_output=flow_output,
    ))
    for key in (
        "prism_frequency_hz",
        "prism_frequency_welch_hz",
        "prism_frequency_lagged_ssm_hz",
    ):
        value = row[key]
        row[key.replace("_hz", "_relative_error")] = (
            abs(float(value) - row["frequency_truth_hz"]) / row["frequency_truth_hz"]
            if value is not None else np.nan
        )
    return row


def analyze_sweep(root , output_csv , **kwargs )  :
    bundles = sorted(path.parent for path in Path(root).rglob("truth.json"))
    if not bundles:
        raise FileNotFoundError(f"no truth.json files found below {root}")
    rows = []
    for index, bundle in enumerate(bundles, 1):
        print(f"[{index}/{len(bundles)}] {bundle.name}", flush=True)
        rows.append(analyze_bundle(bundle, **kwargs))
    frame = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
