"""Motion summaries from confidence-filtered, mask-aligned pose tracks."""

import numpy as np
from scipy.ndimage import median_filter


COCO_TORSO = (5, 6, 11, 12)


def interpolate_short_gaps(xy, confidence, threshold=0.5, max_gap=2):
    """Linearly interpolate bounded joint gaps no longer than max_gap frames."""
    xy = np.asarray(xy, dtype=np.float64).copy()
    confidence = np.asarray(confidence, dtype=np.float64).copy()
    frames, joints = confidence.shape
    observed = np.isfinite(xy).all(axis=2) & np.isfinite(confidence) & (confidence >= threshold)
    interpolated = np.zeros((frames, joints), dtype=bool)

    for joint in range(joints):
        valid = np.flatnonzero(observed[:, joint])
        for left, right in zip(valid[:-1], valid[1:]):
            gap = int(right - left - 1)
            if gap < 1 or gap > max_gap:
                continue
            for offset in range(1, gap + 1):
                weight = offset / float(gap + 1)
                xy[left + offset, joint] = (
                    (1.0 - weight) * xy[left, joint]
                    + weight * xy[right, joint]
                )
                confidence[left + offset, joint] = min(
                    confidence[left, joint], confidence[right, joint]
                )
                interpolated[left + offset, joint] = True

    valid = observed | interpolated
    return xy, confidence, valid, interpolated


def mask_heights(masks, valid_mask=None, smooth_size=5):
    masks = np.asarray(masks, dtype=bool)
    if valid_mask is None:
        valid_mask = np.ones(len(masks), dtype=bool)
    heights = np.full(len(masks), np.nan, dtype=np.float64)
    for index, mask in enumerate(masks):
        if not valid_mask[index] or not mask.any():
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        heights[index] = float(rows[-1] - rows[0] + 1)

    finite = np.isfinite(heights) & (heights > 0)
    if not finite.any():
        return heights
    filled = np.interp(np.arange(len(heights)), np.flatnonzero(finite), heights[finite])
    size = max(1, int(smooth_size))
    if size % 2 == 0:
        size += 1
    return median_filter(filled, size=size, mode="nearest")


def measure_pose_bout(
    xy,
    confidence,
    masks,
    fps,
    core_mask=None,
    mask_valid=None,
    confidence_threshold=0.5,
    maximum_interpolation_gap=2,
    minimum_visible_points=5,
    torso_indices=COCO_TORSO,
    camera_transforms=None,
    camera_translations=None,
):
    """Compute total, torso-translation, and torso-relative pose motion.

    Motion is evaluated only on adjacent frame pairs belonging to the annotated
    core interval. Speeds are expressed in tracked mask heights per second.
    The spatial summary is computed within each frame pair and the temporal
    median is used for the bout summary, matching PRISM's aggregation design.
    """
    xy = np.asarray(xy, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError("xy must have shape (frames, joints, 2)")
    if confidence.shape != xy.shape[:2]:
        raise ValueError("confidence shape must match xy frames and joints")
    if len(masks) != len(xy):
        raise ValueError("masks and pose arrays must have the same frames")

    if core_mask is None:
        core_mask = np.ones(len(xy), dtype=bool)
    if mask_valid is None:
        mask_valid = masks.reshape(len(masks), -1).any(axis=1)
    core_mask = np.asarray(core_mask, dtype=bool)
    mask_valid = np.asarray(mask_valid, dtype=bool)

    original_observed = (
        np.isfinite(xy).all(axis=2)
        & np.isfinite(confidence)
        & (confidence >= confidence_threshold)
    )
    xy, confidence, valid, interpolated = interpolate_short_gaps(
        xy,
        confidence,
        threshold=confidence_threshold,
        max_gap=maximum_interpolation_gap,
    )
    heights = mask_heights(masks, mask_valid)

    total_median = []
    total_p75 = []
    torso_speed = []
    articulated_median = []
    articulated_p75 = []
    expected_pairs = 0
    torso_valid_pairs = 0

    torso_indices = np.asarray(torso_indices, dtype=int)
    torso_indices = torso_indices[torso_indices < xy.shape[1]]
    if camera_transforms is None:
        camera_transforms = [None] * (len(xy) - 1)
    if camera_translations is None:
        camera_translations = [None] * (len(xy) - 1)
    camera_corrected_pairs = 0

    for index in range(len(xy) - 1):
        if not (core_mask[index] and core_mask[index + 1]):
            continue
        expected_pairs += 1
        if not (mask_valid[index] and mask_valid[index + 1]):
            continue
        scale = 0.5 * (heights[index] + heights[index + 1])
        if not np.isfinite(scale) or scale <= 1:
            continue

        keep = valid[index] & valid[index + 1]
        if int(keep.sum()) < int(minimum_visible_points):
            continue
        displacement = xy[index + 1] - xy[index]
        matrix = camera_transforms[index]
        translation = camera_translations[index]
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape == (2, 3) and np.isfinite(matrix).all():
                predicted = xy[index] @ matrix[:, :2].T + matrix[:, 2]
                displacement = displacement - (predicted - xy[index])
                camera_corrected_pairs += 1
        elif translation is not None:
            translation = np.asarray(translation, dtype=np.float64)
            if translation.shape == (2,) and np.isfinite(translation).all():
                displacement = displacement - translation
                camera_corrected_pairs += 1
        multiplier = float(fps) / float(scale)
        speed = np.linalg.norm(displacement[keep], axis=1) * multiplier
        total_median.append(float(np.median(speed)))
        total_p75.append(float(np.percentile(speed, 75)))

        torso_keep = keep[torso_indices]
        if int(torso_keep.sum()) < 2:
            continue
        torso_translation = np.median(
            displacement[torso_indices[torso_keep]], axis=0
        )
        torso_speed.append(float(np.linalg.norm(torso_translation) * multiplier))
        residual = np.linalg.norm(
            displacement[keep] - torso_translation, axis=1
        ) * multiplier
        articulated_median.append(float(np.median(residual)))
        articulated_p75.append(float(np.percentile(residual, 75)))
        torso_valid_pairs += 1

    if expected_pairs == 0 or len(total_median) == 0:
        return {}

    result = {
        "pose_total_motion_median_mask_heights_per_sec": float(np.median(total_median)),
        "pose_total_motion_p75_mask_heights_per_sec": float(np.median(total_p75)),
        "pose_valid_pair_count": len(total_median),
        "pose_expected_pair_count": expected_pairs,
        "pose_valid_pair_fraction": len(total_median) / expected_pairs,
        "pose_interpolated_landmark_fraction": float(interpolated.mean()),
        "pose_observed_landmark_fraction": float(original_observed.mean()),
        "pose_median_mask_height_px": float(np.nanmedian(heights[core_mask])),
        "pose_camera_corrected_pair_count": camera_corrected_pairs,
        "pose_camera_corrected_pair_fraction": camera_corrected_pairs / expected_pairs,
    }
    if torso_valid_pairs:
        result.update({
            "pose_torso_translation_speed_mask_heights_per_sec": float(np.median(torso_speed)),
            "pose_articulated_motion_median_mask_heights_per_sec": float(
                np.median(articulated_median)
            ),
            "pose_articulated_motion_p75_mask_heights_per_sec": float(
                np.median(articulated_p75)
            ),
            "pose_torso_valid_pair_count": torso_valid_pairs,
            "pose_torso_valid_pair_fraction": torso_valid_pairs / expected_pairs,
        })
    return result
