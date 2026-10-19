import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

from .measurements import all_measurements, body_height
from .geometry import (
    add_observation_noise,
    camera_pool,
    dominant_motion_axis,
    load_trajectory,
    project,
    projected_motion_fraction,
    triangulate_sequence,
)


METHODS = (
    "exact_3d",
    "calibrated_multiview_3d",
    "clear_single_view",
    "random_single_view",
    "unfavorable_single_view",
)


def parse_design(condition_id):
    patterns = {
        "frequency_design_hz": r"_f([0-9.]+)",
        "duration_design_sec": r"_T([0-9.]+)",
        "amplitude_design": r"_A([0-9.]+)",
        "simulation_seed": r"_seed([0-9]+)",
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, condition_id)
        values[name] = float(match.group(1)) if match else np.nan
    if np.isfinite(values["simulation_seed"]):
        values["simulation_seed"] = int(values["simulation_seed"])
    values["vigor_design"] = (
        values["frequency_design_hz"] * values["amplitude_design"]
    )
    values["condition_group"] = re.sub(r"_seed[0-9]+.*$", "", condition_id)
    return values


def kinematic_targets(points, fps):
    measured = all_measurements(points, fps)
    return {
        "intensity_target": measured["intensity"],
        "extent_target": measured["extent"],
        "frequency_target_hz": measured["frequency_hz"],
        "periodicity_target": measured["periodicity"],
    }


def select_views(points):
    cameras = camera_pool()
    axis = dominant_motion_axis(points)
    fractions = np.asarray([
        projected_motion_fraction(axis, camera) for camera in cameras
    ])
    return cameras, fractions, int(np.argmax(fractions)), int(np.argmin(fractions))


def observe_views(points, cameras, sigma_px, missing_probability, rng):
    observations = []
    for camera in cameras:
        image, valid = project(points, camera["projection"])
        image[~valid] = np.nan
        observations.append(add_observation_noise(
            image, sigma_px, missing_probability, rng
        ))
    return observations


def measured_row(condition_id, behavior, method, points, fps, **extra):
    return {
        "condition_id": condition_id,
        "behavior": behavior,
        "method": method,
        **extra,
        **all_measurements(points, fps),
    }


def evaluate_condition(points, fps, condition_id, behavior, sigma_px,
                       missing_probability, random_views, seed):
    rng = np.random.default_rng(seed)
    cameras, fractions, clear_index, bad_index = select_views(points)
    observations = observe_views(
        points, cameras, sigma_px, missing_probability, rng
    )
    targets = kinematic_targets(points, fps)
    design = parse_design(condition_id)

    rows = [measured_row(
        condition_id, behavior, "exact_3d", points, fps,
        projection_fraction=1.0,
    )]
    rows.append(measured_row(
        condition_id, behavior, "clear_single_view",
        observations[clear_index], fps,
        camera_index=clear_index,
        projection_fraction=fractions[clear_index],
    ))
    rows.append(measured_row(
        condition_id, behavior, "unfavorable_single_view",
        observations[bad_index], fps,
        camera_index=bad_index,
        projection_fraction=fractions[bad_index],
    ))
    for replicate in range(random_views):
        index = int(rng.integers(0, len(cameras)))
        rows.append(measured_row(
            condition_id, behavior, "random_single_view",
            observations[index], fps,
            random_replicate=replicate,
            camera_index=index,
            projection_fraction=fractions[index],
        ))

    selected = [0, 3, 6, 9]
    stacked = np.stack([observations[index] for index in selected], axis=1)
    reconstruction = triangulate_sequence(
        stacked, [cameras[index]["projection"] for index in selected]
    )
    error = np.linalg.norm(reconstruction - points, axis=2)
    rows.append(measured_row(
        condition_id, behavior, "calibrated_multiview_3d",
        reconstruction, fps,
        reconstruction_rmse=float(np.sqrt(np.nanmean(error ** 2))),
        reconstruction_valid_fraction=float(
            np.isfinite(reconstruction).all(axis=2).mean()
        ),
    ))
    for row in rows:
        row.update(targets)
        row.update(design)
    return rows


def collapse_random_views(frame):
    keys = ["condition_id", "behavior", "method"]
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    collapsed = frame.groupby(keys, as_index=False)[numeric].median()
    text = [
        name for name in frame.columns
        if name not in keys and name not in numeric
    ]
    if text:
        first = frame.groupby(keys, as_index=False)[text].first()
        collapsed = collapsed.merge(first, on=keys, how="left")
    return collapsed


def fit_log_calibration(train_x, train_y, test_x):
    valid = (
        np.isfinite(train_x) & np.isfinite(train_y)
        & (train_x > 0) & (train_y > 0)
    )
    if valid.sum() < 3:
        return np.full(len(test_x), np.nan)
    design = np.column_stack([
        np.ones(valid.sum()), np.log(train_x[valid])
    ])
    coefficients = np.linalg.lstsq(
        design, np.log(train_y[valid]), rcond=None
    )[0]
    prediction = np.full(len(test_x), np.nan)
    eligible = np.isfinite(test_x) & (test_x > 0)
    prediction[eligible] = np.exp(
        coefficients[0] + coefficients[1] * np.log(test_x[eligible])
    )
    return prediction


def cross_validated_calibration(frame):

    frame = collapse_random_views(frame).copy()
    for measure, target in (
        ("intensity", "intensity_target"),
        ("extent", "extent_target"),
    ):
        prediction_name = measure + "_calibrated"
        frame[prediction_name] = np.nan
        for (method, behavior), positions in frame.groupby(
            ["method", "behavior"]
        ).groups.items():
            positions = np.asarray(list(positions))
            block = frame.loc[positions]
            seeds = block.simulation_seed.dropna().unique()
            if len(seeds) < 2:
                continue
            for held_out in seeds:
                test_positions = positions[
                    block.simulation_seed.to_numpy() == held_out
                ]
                train_positions = positions[
                    block.simulation_seed.to_numpy() != held_out
                ]
                frame.loc[test_positions, prediction_name] = fit_log_calibration(
                    frame.loc[train_positions, measure].to_numpy(),
                    frame.loc[train_positions, target].to_numpy(),
                    frame.loc[test_positions, measure].to_numpy(),
                )
        frame[measure + "_relative_error"] = (
            frame[prediction_name] - frame[target]
        ).abs() / frame[target].abs().clip(lower=1e-8)

    frame["frequency_relative_error"] = (
        frame.frequency_hz - frame.frequency_design_hz
    ).abs() / frame.frequency_design_hz.abs().clip(lower=1e-8)
    return frame


def bootstrap_difference(clear, multiview, replicates, rng):
    difference = np.asarray(clear) - np.asarray(multiview)
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return np.nan, np.nan, np.nan
    estimates = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(difference), len(difference))
        estimates[index] = np.median(difference[sample])
    estimate = float(np.median(difference))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return estimate, float(low), float(high)


def construct_summary(frame, output_root, bootstrap_replicates=2000,
                      seed=20261001):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    calibrated = cross_validated_calibration(frame)
    calibrated.to_csv(output_root / "construct_recovery_rows.csv", index=False)

    rows = []
    for (method, behavior), group in calibrated.groupby(["method", "behavior"]):
        for measure, target, error in (
            ("intensity", "vigor_design", "intensity_relative_error"),
            ("extent", "amplitude_design", "extent_relative_error"),
            ("frequency", "frequency_design_hz", "frequency_relative_error"),
        ):
            source = measure if measure != "frequency" else "frequency_hz"
            valid = group[[source, target, error]].dropna()
            rho = spearmanr(valid[source], valid[target]).statistic \
                if len(valid) > 2 else np.nan
            rows.append({
                "method": method,
                "behavior": behavior,
                "measure": measure,
                "n": len(valid),
                "spearman_construct_recovery": rho,
                "median_calibrated_relative_error": valid[error].median(),
                "q90_calibrated_relative_error": valid[error].quantile(0.90),
                "within_10_percent": (valid[error] < 0.10).mean(),
                "within_15_percent": (valid[error] < 0.15).mean(),
                "within_20_percent": (valid[error] < 0.20).mean(),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "construct_recovery_summary.csv", index=False)

    paired_rows = []
    rng = np.random.default_rng(seed)
    for measure, error in (
        ("intensity", "intensity_relative_error"),
        ("extent", "extent_relative_error"),
        ("frequency", "frequency_relative_error"),
    ):
        wide = calibrated.pivot_table(
            index="condition_id", columns="method", values=error, aggfunc="median"
        ).dropna(subset=["clear_single_view", "calibrated_multiview_3d"])
        estimate, low, high = bootstrap_difference(
            wide.clear_single_view,
            wide.calibrated_multiview_3d,
            bootstrap_replicates,
            rng,
        )
        paired_rows.append({
            "measure": measure,
            "n_conditions": len(wide),
            "loss": "relative_error",
            "clear_minus_multiview": estimate,
            "ci_low": low,
            "ci_high": high,
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_root / "paired_view_comparison.csv", index=False)
    return calibrated, summary, paired


def irregular_trajectory(points, fps, level, seed):

    points = np.asarray(points, dtype=float)
    if level <= 0:
        return points.copy()
    rng = np.random.default_rng(seed)
    frames = len(points)
    time = np.arange(frames, dtype=float)
    center = np.nanmedian(points, axis=1, keepdims=True)
    reference = np.nanmedian(points - center, axis=0, keepdims=True)
    deformation = points - center - reference

    increments = np.exp(level * gaussian_filter1d(
        rng.normal(size=frames), sigma=max(1.0, fps / 2.0)
    ))
    warped = np.cumsum(increments)
    warped = (warped - warped[0]) / (warped[-1] - warped[0]) * (frames - 1)
    resampled = np.empty_like(points)
    for joint in range(points.shape[1]):
        for coordinate in range(3):
            resampled[:, joint, coordinate] = np.interp(
                warped, time, points[:, joint, coordinate]
            )

    center = np.nanmedian(resampled, axis=1, keepdims=True)
    reference = np.nanmedian(resampled - center, axis=0, keepdims=True)
    deformation = resampled - center - reference
    envelope = 1.0 + level * gaussian_filter1d(
        rng.normal(size=frames), sigma=max(1.0, fps / 3.0)
    )
    envelope = np.clip(envelope, 0.25, 2.0)
    output = center + reference + envelope[:, None, None] * deformation
    noise = level * 0.01 * body_height(points) * rng.normal(size=output.shape)
    return output + noise


def regularity_rows(input_path, levels, sigma_px=1.0,
                    missing_probability=0.05, seed=20261001,
                    max_files=None):
    input_path = Path(input_path)
    paths = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]
    if max_files is not None and max_files < len(paths):


        selected = np.linspace(0, len(paths) - 1, max_files, dtype=int)
        paths = [paths[index] for index in np.unique(selected)]
    rows = []
    for file_index, path in enumerate(paths, 1):
        print(f"[regularity {file_index}/{len(paths)}] {path.name}", flush=True)
        points, fps, condition_id, behavior = load_trajectory(path)
        for level_index, level in enumerate(levels):
            variant = irregular_trajectory(
                points, fps, level, seed + 1000 * file_index + level_index
            )
            evaluated = evaluate_condition(
                variant, fps,
                condition_id + f"__regularity_{level:g}", behavior,
                sigma_px, missing_probability, 0,
                seed + 2000 * file_index + level_index,
            )
            for row in evaluated:
                if row["method"] in (
                    "exact_3d", "clear_single_view", "calibrated_multiview_3d"
                ):
                    row["irregularity_level"] = level
                    row["base_condition_id"] = condition_id
                    rows.append(row)
    return pd.DataFrame(rows)


def regularity_summary(frame, output_root, bootstrap_replicates=2000,
                       seed=20261001):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "regularity_rows.csv", index=False)
    rows = []
    for (method, behavior), group in frame.groupby(["method", "behavior"]):
        correlations = []
        for _, block in group.groupby("base_condition_id"):
            valid = block[["periodicity", "irregularity_level"]].dropna()
            if len(valid) >= 3 and valid.irregularity_level.nunique() >= 3:
                correlations.append(spearmanr(
                    valid.periodicity, -valid.irregularity_level
                ).statistic)
        rows.append({
            "method": method,
            "behavior": behavior,
            "n_condition_trajectories": len(correlations),
            "median_within_condition_spearman": np.nanmedian(correlations),
            "q25_within_condition_spearman": np.nanquantile(correlations, 0.25),
            "q75_within_condition_spearman": np.nanquantile(correlations, 0.75),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "regularity_construct_summary.csv", index=False)

    condition = frame.pivot_table(
        index=["base_condition_id", "irregularity_level"],
        columns="method", values="periodicity", aggfunc="median"
    ).reset_index()
    correlations = {}
    for method in ("clear_single_view", "calibrated_multiview_3d"):
        values = []
        for _, block in condition.groupby("base_condition_id"):
            valid = block[[method, "irregularity_level"]].dropna()
            if len(valid) >= 3:
                values.append(spearmanr(
                    valid[method], -valid.irregularity_level
                ).statistic)
        correlations[method] = np.asarray(values)
    n = min(len(correlations.get("clear_single_view", [])),
            len(correlations.get("calibrated_multiview_3d", [])))
    rng = np.random.default_rng(seed)
    estimate, low, high = bootstrap_difference(
        1.0 - correlations["clear_single_view"][:n],
        1.0 - correlations["calibrated_multiview_3d"][:n],
        bootstrap_replicates,
        rng,
    )
    paired = pd.DataFrame([{
        "measure": "periodicity_regularization",
        "n_conditions": n,
        "loss": "one_minus_within_condition_spearman",
        "clear_minus_multiview": estimate,
        "ci_low": low,
        "ci_high": high,
    }])
    paired.to_csv(output_root / "regularity_paired_view_comparison.csv", index=False)
    return summary, paired


def run_construct_recovery(input_path, output_csv, summary_root,
                           sigma_px=1.0, missing_probability=0.05,
                           random_views=20, bootstrap_replicates=2000,
                           seed=20261001):
    input_path = Path(input_path)
    paths = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]
    if not paths:
        raise FileNotFoundError(f"no NPZ trajectories found under {input_path}")
    rows = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        points, fps, condition_id, behavior = load_trajectory(path)
        rows.extend(evaluate_condition(
            points, fps, condition_id, behavior,
            sigma_px, missing_probability, random_views, seed + index,
        ))
    raw = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_csv, index=False)
    calibrated, summary, paired = construct_summary(
        raw, summary_root, bootstrap_replicates, seed
    )
    return raw, calibrated, summary, paired
