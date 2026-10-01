"""End-to-end mask-aligned pose baseline and agreement analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism.validation.agreement import agreement_rows

from .inference import UltralyticsPose, infer_bout_pose, read_bout_inputs
from .measurements import measure_pose_bout


def choose_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def run_pose_baseline(
    features_path,
    runs_root,
    output_root,
    videos_root=None,
    model="yolo11m-pose.pt",
    device=None,
    image_size=640,
    padding_fraction=0.15,
    confidence_threshold=0.5,
    maximum_interpolation_gap=2,
    minimum_valid_pair_fraction=0.5,
    bootstrap_replicates=2000,
    seed=20260930,
    force=False,
    limit=None,
):
    features = pd.read_csv(features_path)
    required = {"episode_id", "participant_id", "behavior"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError("features file is missing columns: " + ", ".join(missing))

    runs_root = Path(runs_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    cache_root = output_root / "pose_tracks"
    cache_root.mkdir(parents=True, exist_ok=True)
    estimator = UltralyticsPose(model=model, device=device, image_size=image_size)

    sources = features if limit is None else features.iloc[: int(limit)]
    rows = []
    for count, (_, source) in enumerate(sources.iterrows(), start=1):
        episode_id = str(source["episode_id"])
        cache_path = cache_root / (episode_id + ".npz")
        row = {"episode_id": episode_id}
        try:
            inputs = read_bout_inputs(
                runs_root / episode_id,
                source=source,
                videos_root=videos_root,
            )
            if cache_path.is_file() and not force:
                cached = np.load(cache_path)
                xy = cached["xy"]
                confidence = cached["confidence"]
                overlap = cached["mask_box_iou"]
            else:
                xy, confidence, overlap = infer_bout_pose(
                    inputs, estimator, padding_fraction=padding_fraction
                )
                np.savez_compressed(
                    cache_path,
                    xy=xy,
                    confidence=confidence,
                    mask_box_iou=overlap,
                )

            detected_frames = np.isfinite(confidence).any(axis=1)
            confident_landmarks = (
                np.isfinite(confidence) & (confidence >= confidence_threshold)
            )
            row["pose_detected_frame_count"] = int(detected_frames.sum())
            row["pose_frame_count"] = int(len(confidence))
            row["pose_detected_frame_fraction"] = float(detected_frames.mean())
            row["pose_confident_landmarks_per_frame_median"] = float(
                np.median(confident_landmarks.sum(axis=1))
            )

            timestamps = inputs["timestamps_sec"]
            differences = np.diff(timestamps)
            fps = 1.0 / float(np.median(differences[differences > 0]))
            row.update(measure_pose_bout(
                xy,
                confidence,
                inputs["masks"],
                fps,
                core_mask=inputs["core_mask"],
                mask_valid=inputs["mask_valid"],
                confidence_threshold=confidence_threshold,
                maximum_interpolation_gap=maximum_interpolation_gap,
                camera_transforms=inputs["camera_transforms"],
                camera_translations=inputs["camera_translations"],
                pair_valid=inputs["flow_pair_valid"],
                timestamps=inputs["timestamps_sec"],
            ))
            row["pose_mask_box_iou_median"] = (
                float(np.nanmedian(overlap)) if np.isfinite(overlap).any() else np.nan
            )
            if not row.get("pose_valid_pair_count"):
                row["pose_error"] = "insufficient_valid_pose_pairs"
        except Exception as error:
            row["pose_error"] = type(error).__name__ + ": " + str(error)
        rows.append(row)
        if count % 25 == 0 or count == len(sources):
            print("[" + str(count) + "/" + str(len(sources)) + "] pose bouts processed", flush=True)

    pose = pd.DataFrame(rows)
    merged = features.merge(pose, on="episode_id", how="left", validate="one_to_one")
    # Save inference and diagnostics before any statistical comparison. This
    # guarantees that failed smoke tests remain inspectable.
    pose.to_csv(output_root / "pose_bout_features.csv", index=False)
    merged.to_csv(output_root / "prism_pose_merged.csv", index=False)
    if "pose_valid_pair_fraction" not in merged:
        merged["pose_valid_pair_fraction"] = np.nan
    valid = merged["pose_valid_pair_fraction"].fillna(0) >= minimum_valid_pair_fraction
    analysis = merged.loc[valid].copy()

    total_flow = choose_column(analysis, [
        "prism_intensity_total",
        "optical_flow_pixel_p75_body_heights_per_sec_median_total",
    ])
    residual_flow = choose_column(analysis, [
        "prism_intensity_torso_corrected",
        "prism_intensity",
        "optical_flow_pixel_p75_body_heights_per_sec_median",
    ])
    torso_flow = choose_column(analysis, [
        "prism_torso_translation_speed",
        "optical_flow_torso_translation_speed_body_heights_per_sec_median",
    ])

    comparisons = []
    if total_flow and "pose_total_motion_p75_mask_heights_per_sec" in analysis:
        comparisons.append((
            "pose_total_motion_p75_mask_heights_per_sec",
            total_flow,
            "pose_total_vs_total_flow",
        ))
    if residual_flow and "pose_articulated_motion_p75_mask_heights_per_sec" in analysis:
        comparisons.append((
            "pose_articulated_motion_p75_mask_heights_per_sec",
            residual_flow,
            "pose_articulated_vs_torso_corrected_flow",
        ))
    if torso_flow and "pose_torso_translation_speed_mask_heights_per_sec" in analysis:
        comparisons.append((
            "pose_torso_translation_speed_mask_heights_per_sec",
            torso_flow,
            "pose_torso_translation_vs_flow_torso_translation",
        ))

    agreement = []
    for index, (x, y, name) in enumerate(comparisons):
        agreement.extend(agreement_rows(
            analysis,
            x,
            y,
            name,
            bootstrap_replicates,
            seed + 100 * index,
        ))
    agreement = pd.DataFrame(agreement, columns=[
        "comparison", "x", "y", "analysis", "behavior", "n",
        "n_participants", "correlation", "ci_low", "ci_high",
        "successful_bootstraps",
    ])

    availability = pd.DataFrame([{
        "n_bouts_input": len(features),
        "n_bouts_evaluated": len(pose),
        "n_pose_total_available": int(
            merged.get(
                "pose_total_motion_p75_mask_heights_per_sec", pd.Series(dtype=float)
            ).notna().sum()
        ),
        "n_pose_articulated_available": int(
            merged.get("pose_articulated_motion_p75_mask_heights_per_sec", pd.Series(dtype=float)).notna().sum()
        ),
        "n_pose_torso_available": int(
            merged.get("pose_torso_translation_speed_mask_heights_per_sec", pd.Series(dtype=float)).notna().sum()
        ),
        "n_meeting_valid_pair_threshold": int(valid.sum()),
        "pose_mask_box_iou_median": float(
            merged["pose_mask_box_iou_median"].median()
        ) if "pose_mask_box_iou_median" in merged else np.nan,
        "total_flow_column": total_flow,
        "torso_corrected_flow_column": residual_flow,
        "torso_translation_flow_column": torso_flow,
        "frame_pair_alignment": "intersection_with_frozen_optical_flow_pairs",
        "minimum_valid_pair_fraction": minimum_valid_pair_fraction,
        "confidence_threshold": confidence_threshold,
        "maximum_interpolation_gap": maximum_interpolation_gap,
        "model": model,
        "videos_root": None if videos_root is None else str(videos_root),
    }])

    agreement.to_csv(output_root / "pose_agreement_correlations.csv", index=False)
    availability.to_csv(output_root / "pose_availability.csv", index=False)
    settings = {
        "model": model,
        "videos_root": None if videos_root is None else str(videos_root),
        "device": device,
        "image_size": image_size,
        "padding_fraction": padding_fraction,
        "confidence_threshold": confidence_threshold,
        "maximum_interpolation_gap": maximum_interpolation_gap,
        "minimum_valid_pair_fraction": minimum_valid_pair_fraction,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "comparison_columns": {
            "total_flow": total_flow,
            "torso_corrected_flow": residual_flow,
            "torso_translation_flow": torso_flow,
        },
        "frame_pair_alignment": "intersection_with_frozen_optical_flow_pairs",
    }
    (output_root / "pose_baseline_settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged, agreement, availability
