from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import load_trajectory
from .geometry import (
    add_observation_noise,
    look_at_camera,
    project,
    triangulate_sequence,
)
from .measurements import all_measurements, center_trajectory


MEASURES = ("intensity", "extent", "frequency_hz", "periodicity")


def camera_pool():
    """Predeclared candidate deployment views used for view selection."""
    views = []
    for elevation in (0.0, 20.0):
        for azimuth in np.arange(0.0, 360.0, 30.0):
            views.append(look_at_camera(float(azimuth), elevation))
    return views


def dominant_motion_axis(points):
    centered = center_trajectory(points)
    velocity = np.diff(centered, axis=0).reshape(-1, 3)
    velocity = velocity[np.isfinite(velocity).all(axis=1)]
    covariance = velocity.T @ velocity
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, int(np.argmax(values))]


def projected_motion_fraction(axis, camera):
    """Fraction of the dominant 3D motion axis visible in the image plane."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    viewing_direction = np.asarray(camera["rotation"], dtype=float)[2]
    return float(np.sqrt(max(0.0, 1.0 - np.dot(axis, viewing_direction) ** 2)))


def partial_occlusion(points, fraction=0.35, time_fraction=0.40):
    """Hide the most mobile joints over one contiguous middle interval."""
    output = np.asarray(points, dtype=float).copy()
    mobility = np.nanmean(
        np.sum((output - np.nanmean(output, axis=0, keepdims=True)) ** 2, axis=2),
        axis=0,
    )
    count = max(1, int(np.ceil(fraction * output.shape[1])))
    joints = np.argsort(mobility)[-count:]
    span = max(1, int(round(time_fraction * len(output))))
    start = (len(output) - span) // 2
    output[start:start + span, joints] = np.nan
    return output


def row(condition_id, behavior, category, measurements, **extra):
    return {
        "condition_id": condition_id,
        "behavior": behavior,
        "category": category,
        **extra,
        **measurements,
    }


def evaluate_trajectory(points, fps, condition_id, behavior, sigma_px,
                        missing_probability, random_views, seed):
    cameras = camera_pool()
    axis = dominant_motion_axis(points)
    quality = np.asarray([
        projected_motion_fraction(axis, camera) for camera in cameras
    ])
    clear_index = int(np.argmax(quality))
    unfavorable_index = int(np.argmin(quality))
    rng = np.random.default_rng(seed)

    projected = []
    for camera in cameras:
        image, valid = project(points, camera["projection"])
        image[~valid] = np.nan
        image = add_observation_noise(
            image, sigma_px, missing_probability, rng
        )
        projected.append(image)

    rows = [row(
        condition_id, behavior, "exact_3d",
        all_measurements(points, fps),
        view_selection="ground_truth",
        projection_fraction=1.0,
    )]

    clear_camera = cameras[clear_index]
    rows.append(row(
        condition_id, behavior, "clear_single_view",
        all_measurements(projected[clear_index], fps),
        camera_index=clear_index,
        camera_azimuth_deg=clear_camera["azimuth_deg"],
        camera_elevation_deg=clear_camera["elevation_deg"],
        projection_fraction=quality[clear_index],
        view_selection="maximum_visible_dominant_axis",
    ))

    unfavorable_camera = cameras[unfavorable_index]
    occluded = partial_occlusion(projected[unfavorable_index])
    rows.append(row(
        condition_id, behavior, "unfavorable_occluded_view",
        all_measurements(occluded, fps),
        camera_index=unfavorable_index,
        camera_azimuth_deg=unfavorable_camera["azimuth_deg"],
        camera_elevation_deg=unfavorable_camera["elevation_deg"],
        projection_fraction=quality[unfavorable_index],
        view_selection="minimum_visible_dominant_axis_plus_occlusion",
    ))

    for replicate in range(random_views):
        index = int(rng.integers(0, len(cameras)))
        camera = cameras[index]
        rows.append(row(
            condition_id, behavior, "random_single_view",
            all_measurements(projected[index], fps),
            random_replicate=replicate,
            camera_index=index,
            camera_azimuth_deg=camera["azimuth_deg"],
            camera_elevation_deg=camera["elevation_deg"],
            projection_fraction=quality[index],
            view_selection="uniform_from_predeclared_camera_pool",
        ))

    # Four approximately orthogonal, synchronized, calibrated cameras.
    selected = [0, 3, 6, 9]
    observations = np.stack([projected[index] for index in selected], axis=1)
    projections = [cameras[index]["projection"] for index in selected]
    reconstructed = triangulate_sequence(observations, projections)
    reconstruction_error = np.linalg.norm(reconstructed - points, axis=2)
    rows.append(row(
        condition_id, behavior, "calibrated_multiview_3d",
        all_measurements(reconstructed, fps),
        n_cameras=len(selected),
        camera_indices="-".join(map(str, selected)),
        reconstruction_rmse=float(np.sqrt(np.nanmean(reconstruction_error ** 2))),
        reconstruction_valid_fraction=float(
            np.isfinite(reconstructed).all(axis=2).mean()
        ),
        view_selection="four_predeclared_orthogonal_views",
    ))
    return rows


def add_noninferiority_columns(frame, margins=(0.05, 0.10, 0.15)):
    truth = frame[frame.category.eq("exact_3d")].set_index("condition_id")
    for measure in MEASURES:
        reference = frame.condition_id.map(truth[measure])
        absolute = (frame[measure] - reference).abs()
        minimum_truth = 0.05 if measure == "periodicity" else 1e-8
        eligible = reference.abs() >= minimum_truth
        relative = (absolute / reference.abs().clip(lower=minimum_truth)).where(eligible)
        frame[measure + "_truth"] = reference
        frame[measure + "_absolute_error"] = absolute
        frame[measure + "_relative_error"] = relative
        for margin in margins:
            label = int(round(100 * margin))
            frame[f"{measure}_noninferior_{label}pct"] = relative < margin
    return frame


def run_noninferiority(input_path, output_csv, sigma_px=1.0,
                       missing_probability=0.05, random_views=20,
                       seed=20260930):
    input_path = Path(input_path)
    paths = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]
    if not paths:
        raise FileNotFoundError(f"no NPZ trajectories found under {input_path}")
    rows = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        points, fps, condition_id, behavior = load_trajectory(path)
        rows.extend(evaluate_trajectory(
            points, fps, condition_id, behavior,
            sigma_px, missing_probability, random_views, seed + index,
        ))
    frame = add_noninferiority_columns(pd.DataFrame(rows))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def summarize_noninferiority(frame, output_root, margins=(0.05, 0.10, 0.15)):
    """Aggregate random views within condition before computing pass rates."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    evaluated = frame[~frame.category.eq("exact_3d")].copy()
    condition_rows = []
    for (condition, behavior, category), group in evaluated.groupby(
        ["condition_id", "behavior", "category"], dropna=False
    ):
        result = {
            "condition_id": condition,
            "behavior": behavior,
            "category": category,
            "n_view_realizations": len(group),
        }
        for measure in MEASURES:
            result[measure + "_relative_error"] = group[
                measure + "_relative_error"
            ].median()
        condition_rows.append(result)
    condition_frame = pd.DataFrame(condition_rows)
    condition_frame.to_csv(
        output_root / "condition_level_noninferiority.csv", index=False
    )

    summary_rows = []
    for (category, measure), group in (
        condition_frame.melt(
            id_vars=["condition_id", "behavior", "category"],
            value_vars=[measure + "_relative_error" for measure in MEASURES],
            var_name="measure",
            value_name="relative_error",
        )
        .assign(measure=lambda x: x.measure.str.replace("_relative_error", "", regex=False))
        .groupby(["category", "measure"])
    ):
        values = group.relative_error.dropna()
        base = {
            "category": category,
            "measure": measure,
            "n_conditions": len(values),
            "median_relative_error": float(values.median()),
            "q90_relative_error": float(values.quantile(0.90)),
        }
        for margin in margins:
            label = int(round(100 * margin))
            successes = int((values < margin).sum())
            count = len(values)
            proportion = successes / count if count else np.nan
            # Wilson interval is stable even when every condition passes or fails.
            if count:
                z = 1.959963984540054
                denominator = 1.0 + z * z / count
                center = (proportion + z * z / (2.0 * count)) / denominator
                radius = z * np.sqrt(
                    proportion * (1.0 - proportion) / count
                    + z * z / (4.0 * count * count)
                ) / denominator
                low, high = max(0.0, center - radius), min(1.0, center + radius)
            else:
                low, high = np.nan, np.nan
            base[f"noninferior_fraction_{label}pct"] = proportion
            base[f"noninferior_ci_low_{label}pct"] = low
            base[f"noninferior_ci_high_{label}pct"] = high
        summary_rows.append(base)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "noninferiority_summary.csv", index=False)
    return condition_frame, summary

