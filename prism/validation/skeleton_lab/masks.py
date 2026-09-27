"""Exact synthetic masks and mask geometry."""


import json

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.distance import pdist, squareform


def render_exact_sequence(
    bundle ,
    flow_variant  = "skeleton",
    mask_variant  = "simple",
    max_dimension  = 256,
    remove_camera_motion  = False,
)    :
    """Recreate textured frames and exact body masks from one simulator state.

    Skeleton Lab deliberately makes its capsule/realistic renderings nearly
    texture-free. Dense optical flow is underdetermined inside such constant
    regions, so the default uses the high-contrast skeleton rendering for the
    flow image and the filled capsule rendering for the exact body mask. Both
    renderers use the identical simulated motion.
    """
    from skeleton_lab.render import Renderer
    from skeleton_lab.simulate import simulate
    from skeleton_lab.spec import ScenarioSpec

    payload = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    if remove_camera_motion:
        # Noise streams are category-keyed in Skeleton Lab, so disabling only
        # camera motion preserves the identical physical movement realization.
        payload["camera"]["shake"]["sigma_px"] = 0.0
        payload["camera"]["pan_dx"] = {"type": "const", "value": 0.0}
        payload["camera"]["pan_dy"] = {"type": "const", "value": 0.0}
        payload["camera"]["zoom"] = {"type": "const", "value": 1.0}
    spec = ScenarioSpec.model_validate(payload)
    result = simulate(spec)
    flow_renderer = Renderer(result, variant=flow_variant)
    mask_renderer = Renderer(result, variant=mask_variant)
    frames  = []
    masks  = []
    for index in range(result.n_frames):
        frame = flow_renderer.frame(index)
        mask_frame = mask_renderer.frame(index)
        # The renderer background is exact. Anti-aliased foreground pixels still
        # differ from it and are retained by the per-channel maximum.
        difference = cv2.absdiff(mask_frame, mask_renderer.background)
        mask = np.max(difference, axis=2) > 1
        if max(frame.shape[:2]) > max_dimension:
            scale = max_dimension / max(frame.shape[:2])
            size = (max(2, int(round(frame.shape[1] * scale))),
                    max(2, int(round(frame.shape[0] * scale))))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        frames.append(frame)
        masks.append(mask)
    return np.asarray(frames), np.asarray(masks, dtype=bool), float(spec.fps)


def mask_height(mask )  :
    rows = np.flatnonzero(np.any(mask, axis=1))
    return float(rows[-1] - rows[0] + 1) if rows.size else np.nan


def mask_distance_matrix(masks )  :
    """Area-normalized Euclidean distances between binary masks."""
    flat = np.asarray(masks, dtype=bool).reshape(len(masks), -1).astype(np.float32)
    areas = flat.sum(axis=1, dtype=np.float64)
    gram = np.asarray(flat @ flat.T, dtype=np.float64)
    squared = np.maximum(areas[:, None] + areas[None, :] - 2.0 * gram, 0.0)
    return np.sqrt(squared / max(float(np.median(areas)), 1.0))


def temporal_majority_masks(masks , width  = 3)  :
    """Remove isolated temporal mask flicker with a short majority filter."""
    if width < 1 or width % 2 == 0:
        raise ValueError("temporal filter width must be a positive odd integer")
    values = median_filter(
        np.asarray(masks, dtype=np.uint8), size=(width, 1, 1), mode="nearest"
    )
    return values > 0


def resize_mask_sequence(masks , max_dimension  = 64)  :
    """Resize masks for a resolution-controlled representation ablation."""
    masks = np.asarray(masks, dtype=bool)
    height, width = masks.shape[1:]
    if max(height, width) <= max_dimension:
        return masks.copy()
    scale = max_dimension / max(height, width)
    size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    return np.asarray([
        cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        for mask in masks
    ])


def signed_distance_sequence(
    masks ,
    truncation_body_heights  = 0.15,
)  :
    """Body-height-normalized truncated signed-distance fields.

    Positive values are inside the silhouette and negative values are outside.
    A single robust sequence-level mask height keeps the spatial scale fixed
    within a bout, avoiding framewise denominator feedback.
    """
    masks = np.asarray(masks, dtype=bool)
    heights = np.asarray([mask_height(mask) for mask in masks], dtype=float)
    reference_height = float(np.nanmedian(heights))
    if not np.isfinite(reference_height) or reference_height <= 0:
        raise ValueError("cannot construct signed distances from empty masks")
    fields = []
    for mask in masks:
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        field = (inside - outside) / reference_height
        fields.append(np.clip(field, -truncation_body_heights,
                              truncation_body_heights))
    return np.asarray(fields, dtype=np.float32)


def representation_sequence(
    masks ,
    method ,
    *,
    temporal_width  = 3,
    truncation_body_heights  = 0.15,
    max_dimension  = 64,
)  :
    """Construct one of the four pre-registered mask ablation inputs."""
    resized = resize_mask_sequence(masks, max_dimension=max_dimension)
    if method == "binary_euclidean":
        return resized.astype(np.float32)
    if method == "temporal_binary_euclidean":
        return temporal_majority_masks(resized, temporal_width).astype(np.float32)
    if method in {"tsdf_euclidean", "tsdf_huber"}:
        return signed_distance_sequence(
            resized, truncation_body_heights=truncation_body_heights
        )
    raise ValueError(f"unknown mask representation method: {method}")


def representation_distance_matrix(
    sequence ,
    metric  = "euclidean",
    huber_delta  = 0.02,
    block_size  = 32,
)  :
    """Resolution-normalized frame dissimilarities for the ablation.

    Euclidean values are root-mean-square feature differences. Huber values
    use ``sqrt(2 * mean(Huber_delta(x-y)))``, which has the same local scale as
    RMS distance but limits the influence of isolated large residuals.
    """
    flat = np.asarray(sequence, dtype=np.float32).reshape(len(sequence), -1)
    if metric == "euclidean":
        return squareform(pdist(flat, metric="euclidean")) / np.sqrt(flat.shape[1])
    if metric != "huber":
        raise ValueError("metric must be 'euclidean' or 'huber'")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    count = len(flat)
    result = np.zeros((count, count), dtype=np.float32)
    # Blocked broadcasting avoids constructing an n x n x pixels tensor.
    for first in range(0, count, block_size):
        left = flat[first:first + block_size]
        for second in range(first, count, block_size):
            right = flat[second:second + block_size]
            residual = np.abs(left[:, None, :] - right[None, :, :])
            loss = np.where(
                residual <= huber_delta,
                0.5 * residual * residual,
                huber_delta * (residual - 0.5 * huber_delta),
            )
            values = np.sqrt(2.0 * loss.mean(axis=2))
            result[first:first + len(left), second:second + len(right)] = values
            result[second:second + len(right), first:first + len(left)] = values.T
    np.fill_diagonal(result, 0.0)
    return np.asarray(result, dtype=np.float64)
