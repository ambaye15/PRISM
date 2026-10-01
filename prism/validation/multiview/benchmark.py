from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import (
    add_observation_noise,
    look_at_camera,
    project,
    triangulate_sequence,
)
from .measurements import all_measurements


DEFAULT_VIEWS = (
    (0.0, 10.0),
    (90.0, 10.0),
    (180.0, 10.0),
    (270.0, 10.0),
)


def load_trajectory(path):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    joint_key = next(
        (key for key in ("joints_3d", "keypoints_3d", "world_joints") if key in keys),
        None,
    )
    if joint_key is None:
        raise ValueError(f"{path} has no supported 3D joint array")
    joints = np.asarray(data[joint_key], dtype=float)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError("3D joints must have shape (frames, joints, 3)")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    condition_id = str(np.asarray(data["condition_id"]).reshape(-1)[0]) \
        if "condition_id" in keys else Path(path).stem
    behavior = str(np.asarray(data["behavior"]).reshape(-1)[0]) \
        if "behavior" in keys else "unknown"
    return joints, fps, condition_id, behavior


def measurement_rows(points, fps, condition_id, behavior, sigma_px,
                     missing_probability, seed):
    cameras = [look_at_camera(*view) for view in DEFAULT_VIEWS]
    projections = [camera["projection"] for camera in cameras]
    rng = np.random.default_rng(seed)
    observations = []
    rows = []

    truth = all_measurements(points, fps)
    rows.append({
        "condition_id": condition_id,
        "behavior": behavior,
        "method": "exact_3d",
        "n_cameras": 0,
        "camera_ids": "truth",
        **truth,
    })

    for index, (camera, projection) in enumerate(zip(cameras, projections)):
        image, valid = project(points, projection)
        image[~valid] = np.nan
        image = add_observation_noise(image, sigma_px, missing_probability, rng)
        observations.append(image)
        measured = all_measurements(image, fps)
        rows.append({
            "condition_id": condition_id,
            "behavior": behavior,
            "method": "single_view_2d",
            "n_cameras": 1,
            "camera_ids": str(index),
            "camera_azimuth_deg": camera["azimuth_deg"],
            "camera_elevation_deg": camera["elevation_deg"],
            **measured,
        })

    observations = np.stack(observations, axis=1)
    for count in (2, 3, 4):
        for selected in combinations(range(len(cameras)), count):
            selected = tuple(selected)
            reconstructed = triangulate_sequence(
                observations[:, selected],
                [projections[index] for index in selected],
            )
            measured = all_measurements(reconstructed, fps)
            reconstruction_error = np.linalg.norm(reconstructed - points, axis=2)
            rows.append({
                "condition_id": condition_id,
                "behavior": behavior,
                "method": "triangulated_3d",
                "n_cameras": count,
                "camera_ids": "-".join(map(str, selected)),
                "reconstruction_rmse": float(np.sqrt(np.nanmean(reconstruction_error ** 2))),
                "reconstruction_valid_fraction": float(np.isfinite(reconstructed).all(axis=2).mean()),
                **measured,
            })
    return rows


def add_errors(frame):
    measures = ("intensity", "extent", "frequency_hz", "periodicity")
    truth = frame[frame.method.eq("exact_3d")].set_index("condition_id")
    for measure in measures:
        mapping = truth[measure]
        reference = frame.condition_id.map(mapping)
        difference = (frame[measure] - reference).abs()
        frame[measure + "_absolute_error"] = difference
        if measure != "periodicity":
            frame[measure + "_relative_error"] = difference / reference.abs().clip(lower=1e-12)
    return frame


def run_benchmark(input_path, output_csv, sigma_px=1.0,
                  missing_probability=0.05, seed=20260929):
    input_path = Path(input_path)
    paths = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]
    if not paths:
        raise FileNotFoundError(f"no NPZ trajectories found under {input_path}")
    rows = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        points, fps, condition_id, behavior = load_trajectory(path)
        rows.extend(measurement_rows(
            points, fps, condition_id, behavior,
            sigma_px, missing_probability, seed + index,
        ))
    frame = add_errors(pd.DataFrame(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame

