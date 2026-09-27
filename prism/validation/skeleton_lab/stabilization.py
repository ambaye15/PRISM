"""Camera-motion estimation and joint frame/mask stabilization."""


import cv2
import numpy as np


class StabilizationDiagnostics:
    """Quality-control values from background camera registration."""

    __slots__ = (
        "method",
        "valid_pair_fraction",
        "median_inliers",
        "maximum_translation_px",
        "maximum_rotation_deg",
        "maximum_scale_change",
    )

    def __init__(
        self,
        method,
        valid_pair_fraction,
        median_inliers,
        maximum_translation_px,
        maximum_rotation_deg,
        maximum_scale_change,
    ):
        self.method = method
        self.valid_pair_fraction = valid_pair_fraction
        self.median_inliers = median_inliers
        self.maximum_translation_px = maximum_translation_px
        self.maximum_rotation_deg = maximum_rotation_deg
        self.maximum_scale_change = maximum_scale_change


def _homogeneous(matrix )  :
    result = np.eye(3, dtype=np.float64)
    result[:2] = np.asarray(matrix, dtype=np.float64)
    return result


def estimate_background_warps(
    frames ,
    masks ,
    *,
    exclusion_dilation_px  = 8,
    max_corners  = 600,
    quality_level  = 0.01,
    min_distance_px  = 7.0,
    min_inliers  = 12,
    ransac_threshold_px  = 2.0,
    max_translation_px  = 40.0,
    max_rotation_deg  = 8.0,
    max_scale_change  = 0.08,
)   :
    """Estimate warps mapping each frame into frame-zero coordinates.

    Features are detected only outside a dilated union of the subject masks.
    Pairwise similarity transforms are estimated with RANSAC and accumulated.
    Unreliable pairs use the identity transform rather than subject motion.
    """
    frames = np.asarray(frames)
    masks = np.asarray(masks, dtype=bool)
    if len(frames) != len(masks) or len(frames) < 2:
        raise ValueError("frames and masks must have the same length >= 2")

    grays = np.asarray([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    warps = [np.eye(3, dtype=np.float64)]
    valid  = []
    inlier_counts  = []
    translations  = []
    rotations  = []
    scales  = []

    for index in range(len(frames) - 1):
        excluded = masks[index] | masks[index + 1]
        if exclusion_dilation_px > 0:
            excluded = cv2.dilate(
                excluded.astype(np.uint8), kernel, iterations=exclusion_dilation_px
            ).astype(bool)
        feature_mask = (~excluded).astype(np.uint8) * 255
        points = cv2.goodFeaturesToTrack(
            grays[index], maxCorners=max_corners, qualityLevel=quality_level,
            minDistance=min_distance_px, mask=feature_mask,
        )
        accepted = False
        pair = np.eye(3, dtype=np.float64)
        count = 0
        translation = rotation = scale_change = 0.0
        if points is not None and len(points) >= min_inliers:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                grays[index], grays[index + 1], points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if tracked is not None and status is not None:
                keep = status.reshape(-1).astype(bool)
                source = points.reshape(-1, 2)[keep]
                target = tracked.reshape(-1, 2)[keep]
                if len(source) >= min_inliers:
                    affine, inliers = cv2.estimateAffinePartial2D(
                        source, target, method=cv2.RANSAC,
                        ransacReprojThreshold=ransac_threshold_px,
                        maxIters=2000, confidence=0.995, refineIters=10,
                    )
                    if affine is not None and inliers is not None:
                        count = int(inliers.sum())
                        a, b = float(affine[0, 0]), float(affine[1, 0])
                        scale = float(np.hypot(a, b))
                        rotation = float(np.degrees(np.arctan2(b, a)))
                        translation = float(np.hypot(affine[0, 2], affine[1, 2]))
                        scale_change = abs(scale - 1.0)
                        accepted = (
                            count >= min_inliers
                            and translation <= max_translation_px
                            and abs(rotation) <= max_rotation_deg
                            and scale_change <= max_scale_change
                        )
                        if accepted:
                            pair = _homogeneous(affine)
        valid.append(accepted)
        inlier_counts.append(count)
        translations.append(translation)
        rotations.append(abs(rotation))
        scales.append(scale_change)
        # pair maps frame i -> i+1. Invert it, then compose with the
        # already-known mapping from frame i -> reference frame zero.
        warps.append(warps[-1] @ np.linalg.inv(pair))

    diagnostics = StabilizationDiagnostics(
        method="background_similarity_ransac",
        valid_pair_fraction=float(np.mean(valid)),
        median_inliers=float(np.median(inlier_counts)),
        maximum_translation_px=float(np.max(translations)),
        maximum_rotation_deg=float(np.max(rotations)),
        maximum_scale_change=float(np.max(scales)),
    )
    return np.asarray(warps), diagnostics


def stabilize_sequence(
    frames ,
    masks ,
    warps_to_reference ,
)   :
    """Apply the same source-to-reference affine warps to frames and masks."""
    frames = np.asarray(frames)
    masks = np.asarray(masks, dtype=bool)
    warps = np.asarray(warps_to_reference, dtype=np.float64)
    if len(frames) != len(masks) or len(frames) != len(warps):
        raise ValueError("frames, masks, and warps must have identical lengths")
    height, width = frames.shape[1:3]
    corrected_frames = []
    corrected_masks = []
    for frame, mask, warp in zip(frames, masks, warps):
        matrix = warp[:2]
        corrected_frames.append(cv2.warpAffine(
            frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ))
        corrected_masks.append(cv2.warpAffine(
            mask.astype(np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0)
    return np.asarray(corrected_frames), np.asarray(corrected_masks, dtype=bool)
