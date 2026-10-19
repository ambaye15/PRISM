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
    estimator = None

    sources = features if limit is None else features.iloc[: int(limit)]
    rows = []
    for count, (_, source) in enumerate(sources.iterrows(), start=1):
        episode_id = str(source["episode_id"])
        cache_path = cache_root / (episode_id + ".npz")
        cache_metadata = cache_path.with_suffix('.json')
        expected_cache = {"model":model,"image_size":image_size,"padding_fraction":padding_fraction}
        use_cache = cache_path.is_file() and not force
        if use_cache and cache_metadata.is_file():
            use_cache = all(json.loads(cache_metadata.read_text()).get(key) == value for key, value in expected_cache.items())
        row = {"episode_id": episode_id,
               "pose_cache_status": "reused" if use_cache else "new"}

        try:
            inputs = read_bout_inputs(
                runs_root / episode_id,
                source=source,
                videos_root=videos_root,
                require_video=not use_cache,
            )
            if use_cache:
                cached = np.load(cache_path)
                xy = cached["xy"]
                confidence = cached["confidence"]
                overlap = cached["mask_box_iou"]
            else:
                if estimator is None:
                    estimator = UltralyticsPose(model=model, device=device, image_size=image_size)
                xy, confidence, overlap = infer_bout_pose(
                    inputs, estimator, padding_fraction=padding_fraction
                )
                np.savez_compressed(
                    cache_path,
                    xy=xy,
                    confidence=confidence,
                    mask_box_iou=overlap,
                )
                cache_metadata.write_text(json.dumps(expected_cache,indent=2)+'\n')
                row['pose_cache_status']='new'


            if len(xy) != len(inputs['masks']) or xy.shape[1:] != (17,2):
                raise ValueError('Cached pose shape differs from bout masks/COCO-17; rerun with --force')
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
            pairs = pd.read_csv(runs_root / episode_id / "optical_flow_frames.csv")
            height_by_pair = np.full(len(xy) - 1, np.nan)
            for _, pair in pairs.iterrows():
                i = int(pair["frame_index"])
                if 0 <= i < len(height_by_pair):
                    height_by_pair[i] = pair.get("body_height_pixels", np.nan)
            pose_result = measure_pose_bout(
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
                pair_heights=height_by_pair,
                return_pair_indices=True,
            )
            total_indices = pose_result.pop("total_pair_indices", [])
            torso_indices = pose_result.pop("torso_pair_indices", [])
            pose_values = {"flow_total_shared_pose_pairs": pose_result.pop("total_pair_values", []),
                           "flow_residual_shared_pose_pairs": pose_result.pop("articulated_pair_values", []),
                           "flow_torso_shared_pose_pairs": pose_result.pop("torso_pair_values", [])}
            row.update(pose_result)
            specifications = [
                ("flow_total_shared_pose_pairs", "flow_pixel_p75_body_heights_per_sec", total_indices),
                ("flow_residual_shared_pose_pairs", "flow_pixel_p75_torso_corrected_body_heights_per_sec", torso_indices),
                ("flow_torso_shared_pose_pairs", "flow_torso_translation_body_heights_per_sec", torso_indices),
            ]
            for target, column, indices in specifications:
                if column not in pairs:
                    raise ValueError("Missing per-pair flow column: " + column)
                ordered_pairs = pairs.set_index("frame_index").reindex(indices)
                values = pd.to_numeric(ordered_pairs[column], errors="coerce")
                valid_shared = np.isfinite(values.to_numpy())
                pose_column = {"flow_total_shared_pose_pairs": "pose_total_motion_p75_mask_heights_per_sec",
                               "flow_residual_shared_pose_pairs": "pose_articulated_motion_p75_mask_heights_per_sec",
                               "flow_torso_shared_pose_pairs": "pose_torso_translation_speed_mask_heights_per_sec"}[target]
                matching_pose = np.asarray(pose_values[target], dtype=float)
                row[pose_column] = float(np.median(matching_pose[valid_shared])) if valid_shared.any() else np.nan
                values = values.loc[valid_shared]
                row[target] = float(values.median()) if len(values) else np.nan
                row[target + "_n"] = len(values)

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


    pose.to_csv(output_root / "pose_bout_features.csv", index=False)
    merged.to_csv(output_root / "prism_pose_merged.csv", index=False)
    if "pose_valid_pair_fraction" not in merged:
        merged["pose_valid_pair_fraction"] = np.nan
    valid = merged["pose_valid_pair_fraction"].fillna(0) >= minimum_valid_pair_fraction
    analysis = merged.loc[valid].copy()

    total_flow = "flow_total_shared_pose_pairs"
    residual_flow = "flow_residual_shared_pose_pairs"
    torso_flow = "flow_torso_shared_pose_pairs"

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

    if not comparisons:
        raise ValueError("No paired pose metrics available; inspect pose_bout_features.csv and pose_error.")
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
        "frame_pair_alignment": "exact_comparison_specific_intersection; frozen_flow_pair_height",
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
        "frame_pair_alignment": "exact_comparison_specific_intersection; frozen_flow_pair_height",
    }
    (output_root / "pose_baseline_settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged, agreement, availability
