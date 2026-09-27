"""Subject-masked optical-flow measurements for annotated SMM bouts."""

import csv
import json
from pathlib import Path

import numpy as np

from ._records import asdict, field, record


@record
class OpticalFlowSettings:
    """Configuration for body-scale-normalized visual motion measurements."""

    backend = field("dis")
    device = field("auto")
    camera_correction = field("ransac_similarity")
    region_mode = field("flow_consistent")
    erosion_body_fraction = field(0.005)
    target_dilation_body_fraction = field(0.020)
    active_threshold_body_heights_per_sec = field(0.05)
    minimum_valid_pixels = field(100)
    minimum_retained_pixel_fraction = field(0.20)
    minimum_valid_pair_fraction = field(0.80)
    maximum_flow_dimension = field(768)
    body_height_smoothing_window = field(5)
    forward_backward_check = field(True)
    forward_backward_threshold_body_fraction = field(0.02)
    ransac_subject_exclusion_body_fraction = field(0.05)
    ransac_max_corners = field(1000)
    ransac_quality_level = field(0.01)
    ransac_min_distance_pixels = field(8.0)
    ransac_reprojection_threshold_pixels = field(3.0)
    ransac_min_matches = field(30)
    ransac_min_inlier_fraction = field(0.35)
    torso_correction = field(True)
    torso_radius_x_fraction = field(0.18)
    torso_radius_y_fraction = field(0.22)
    torso_minimum_pixels = field(25)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _json_value(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("Cannot serialize %s" % type(value).__name__)


def _safe_summary(values, statistic):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    if statistic == "median":
        return float(np.median(array))
    if statistic == "p90":
        return float(np.percentile(array, 90))
    if statistic == "mean":
        return float(np.mean(array))
    raise ValueError("Unknown statistic: %s" % statistic)


def _mask_height(mask):
    rows = np.flatnonzero(np.any(mask, axis=1))
    if not len(rows):
        return None
    return float(rows[-1] - rows[0] + 1)


def _morphology_iterations(body_height, fraction):
    if fraction <= 0:
        return 0
    return max(1, int(round(float(body_height) * float(fraction))))


def _erode(mask, iterations):
    if iterations <= 0:
        return mask.astype(bool, copy=True)
    from scipy.ndimage import binary_erosion

    result = binary_erosion(
        mask.astype(bool),
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(iterations),
        border_value=0,
    )
    return result.astype(bool)


def _dilate(mask, iterations):
    if iterations <= 0:
        return mask.astype(bool, copy=True)
    from scipy.ndimage import binary_dilation

    result = binary_dilation(
        mask.astype(bool),
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(iterations),
        border_value=0,
    )
    return result.astype(bool)


def _smooth_mask_heights(masks, valid, window):
    """Return a robust body-scale trajectory without framewise mask jitter."""
    raw = np.asarray(
        [
            _mask_height(mask) if bool(is_valid) else np.nan
            for mask, is_valid in zip(masks, valid)
        ],
        dtype=np.float64,
    )
    finite = np.flatnonzero(np.isfinite(raw))
    if not len(finite):
        return raw
    missing = np.flatnonzero(~np.isfinite(raw))
    if len(missing):
        raw[missing] = np.interp(missing, finite, raw[finite])
    width = max(1, int(window))
    if width % 2 == 0:
        width += 1
    if width == 1:
        return raw
    radius = width // 2
    padded = np.pad(raw, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + width]) for index in range(len(raw))],
        dtype=np.float64,
    )


def _resize_for_flow(frame, maximum_dimension):
    import cv2

    height, width = frame.shape[:2]
    if maximum_dimension is None or maximum_dimension <= 0:
        return frame, 1.0, 1.0
    scale = min(1.0, float(maximum_dimension) / max(height, width))
    target_height = max(8, int(round(height * scale / 8.0)) * 8)
    target_width = max(8, int(round(width * scale / 8.0)) * 8)
    if target_height == height and target_width == width:
        return frame, 1.0, 1.0
    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, target_width / width, target_height / height


def _restore_flow_resolution(flow, output_shape, scale_x, scale_y):
    import cv2

    output_height, output_width = output_shape
    if flow.shape[:2] == (output_height, output_width):
        return flow.astype(np.float32, copy=False)
    restored = cv2.resize(
        flow,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    restored[..., 0] /= max(float(scale_x), 1e-12)
    restored[..., 1] /= max(float(scale_y), 1e-12)
    return restored


class OpenCvFlowEstimator:
    """Dense optical flow using algorithms distributed with OpenCV."""

    def __init__(self, backend, maximum_dimension):
        import cv2

        self.backend = backend
        self.maximum_dimension = maximum_dimension
        self.estimator = None
        if backend == "dis":
            self.estimator = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            )
            self.estimator.setUseSpatialPropagation(True)
        elif backend != "farneback":
            raise ValueError("OpenCV backend must be 'dis' or 'farneback'")

    def estimate(self, previous_frame, current_frame):
        import cv2

        previous, scale_x, scale_y = _resize_for_flow(
            previous_frame, self.maximum_dimension
        )
        current, _, _ = _resize_for_flow(current_frame, self.maximum_dimension)
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        if self.backend == "dis":
            flow = self.estimator.calc(previous_gray, current_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                current_gray,
                None,
                0.5,
                4,
                25,
                5,
                7,
                1.5,
                0,
            )
        return _restore_flow_resolution(
            flow,
            previous_frame.shape[:2],
            scale_x,
            scale_y,
        )


class RaftFlowEstimator:
    """Optional torchvision RAFT estimator for the paper-quality comparison."""

    def __init__(self, backend, device, maximum_dimension):
        import torch
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        self.torch = torch
        self.maximum_dimension = maximum_dimension
        self.device = _choose_torch_device(device)
        if backend == "raft_small":
            self.weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=self.weights, progress=True)
        elif backend == "raft_large":
            self.weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=self.weights, progress=True)
        else:
            raise ValueError("RAFT backend must be 'raft_small' or 'raft_large'")
        self.model = self.model.eval().to(self.device)
        self.transforms = self.weights.transforms()

    def estimate(self, previous_frame, current_frame):
        import cv2

        previous, scale_x, scale_y = _resize_for_flow(
            previous_frame, self.maximum_dimension
        )
        current, _, _ = _resize_for_flow(current_frame, self.maximum_dimension)
        previous_rgb = cv2.cvtColor(previous, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
        previous_tensor = self.torch.from_numpy(
            np.ascontiguousarray(previous_rgb)
        ).permute(2, 0, 1)[None]
        current_tensor = self.torch.from_numpy(
            np.ascontiguousarray(current_rgb)
        ).permute(2, 0, 1)[None]
        previous_tensor, current_tensor = self.transforms(
            previous_tensor, current_tensor
        )
        with self.torch.inference_mode():
            prediction = self.model(
                previous_tensor.to(self.device),
                current_tensor.to(self.device),
            )[-1][0]
        flow = prediction.detach().to("cpu").permute(1, 2, 0).numpy()
        return _restore_flow_resolution(
            flow,
            previous_frame.shape[:2],
            scale_x,
            scale_y,
        )


def _choose_torch_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_flow_estimator(settings):
    if settings.backend in {"dis", "farneback"}:
        return OpenCvFlowEstimator(
            settings.backend,
            settings.maximum_flow_dimension,
        )
    if settings.backend in {"raft_small", "raft_large"}:
        return RaftFlowEstimator(
            settings.backend,
            settings.device,
            settings.maximum_flow_dimension,
        )
    raise ValueError(
        "backend must be one of: dis, farneback, raft_small, raft_large"
    )


def _background_translation(flow, previous_mask, current_mask, body_height):
    exclusion_iterations = _morphology_iterations(body_height, 0.05)
    excluded = _dilate(
        np.logical_or(previous_mask, current_mask),
        exclusion_iterations,
    )
    background = np.logical_not(excluded)
    finite = np.isfinite(flow).all(axis=2)
    usable = np.logical_and(background, finite)
    if int(usable.sum()) < 100:
        return np.zeros(2, dtype=np.float32), 0
    vectors = flow[usable].astype(np.float64)
    center = np.median(vectors, axis=0)
    residual = np.linalg.norm(vectors - center[None], axis=1)
    cutoff = np.percentile(residual, 90)
    inliers = residual <= cutoff
    if int(inliers.sum()) >= 100:
        center = np.median(vectors[inliers], axis=0)
    return center.astype(np.float32), int(inliers.sum())


def _dense_transform_flow(matrix, shape):
    """Convert a 2x3 image transform into its dense displacement field."""
    height, width = shape
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    transformed_x = (
        matrix[0, 0] * x_grid + matrix[0, 1] * y_grid + matrix[0, 2]
    )
    transformed_y = (
        matrix[1, 0] * x_grid + matrix[1, 1] * y_grid + matrix[1, 2]
    )
    return np.stack(
        (transformed_x - x_grid, transformed_y - y_grid), axis=-1
    ).astype(np.float32)


def _ransac_camera_flow(
    previous_frame,
    current_frame,
    previous_mask,
    current_mask,
    body_height,
    settings,
):
    """Estimate global background motion using tracked features and RANSAC."""
    import cv2

    exclusion = _morphology_iterations(
        body_height, settings.ransac_subject_exclusion_body_fraction
    )
    excluded_previous = _dilate(previous_mask, exclusion)
    excluded_current = _dilate(current_mask, exclusion)
    feature_mask = (255 * np.logical_not(excluded_previous)).astype(np.uint8)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    points_previous = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=int(settings.ransac_max_corners),
        qualityLevel=float(settings.ransac_quality_level),
        minDistance=float(settings.ransac_min_distance_pixels),
        mask=feature_mask,
    )
    detected = 0 if points_previous is None else len(points_previous)
    diagnostics = {
        "flow_camera_model_requested": settings.camera_correction,
        "flow_camera_feature_count": detected,
        "flow_camera_match_count": 0,
        "flow_camera_inlier_count": 0,
        "flow_camera_inlier_fraction": 0.0,
        "flow_camera_reprojection_median_pixels": None,
        "flow_camera_ransac_success": False,
    }
    if points_previous is None or detected < int(settings.ransac_min_matches):
        return None, diagnostics
    points_current, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points_previous,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if points_current is None or status is None:
        return None, diagnostics
    source = points_previous.reshape(-1, 2)
    target = points_current.reshape(-1, 2)
    keep = status.reshape(-1).astype(bool)
    keep &= np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    height, width = previous_mask.shape
    rounded = np.rint(target).astype(np.int64)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    keep &= inside
    valid_indices = np.flatnonzero(keep)
    if len(valid_indices):
        target_pixels = rounded[valid_indices]
        outside_subject = np.logical_not(
            excluded_current[target_pixels[:, 1], target_pixels[:, 0]]
        )
        keep[valid_indices] &= outside_subject
    source = source[keep]
    target = target[keep]
    diagnostics["flow_camera_match_count"] = len(source)
    if len(source) < int(settings.ransac_min_matches):
        return None, diagnostics
    if settings.camera_correction == "ransac_similarity":
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(
                settings.ransac_reprojection_threshold_pixels
            ),
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
    elif settings.camera_correction == "ransac_affine":
        matrix, inlier_mask = cv2.estimateAffine2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(
                settings.ransac_reprojection_threshold_pixels
            ),
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
    else:
        raise ValueError("RANSAC correction requires similarity or affine mode")
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return None, diagnostics
    inliers = inlier_mask.reshape(-1).astype(bool)
    n_inliers = int(inliers.sum())
    inlier_fraction = float(n_inliers / len(source)) if len(source) else 0.0
    diagnostics["flow_camera_inlier_count"] = n_inliers
    diagnostics["flow_camera_inlier_fraction"] = inlier_fraction
    if n_inliers:
        predicted = source @ matrix[:, :2].T + matrix[:, 2]
        residual = np.linalg.norm(predicted - target, axis=1)
        diagnostics["flow_camera_reprojection_median_pixels"] = float(
            np.median(residual[inliers])
        )
    success = bool(
        n_inliers >= int(settings.ransac_min_matches)
        and inlier_fraction >= float(settings.ransac_min_inlier_fraction)
    )
    diagnostics["flow_camera_ransac_success"] = success
    if not success:
        return None, diagnostics
    diagnostics["flow_camera_transform"] = matrix.tolist()
    return _dense_transform_flow(matrix, previous_mask.shape), diagnostics


def _residual_background_rate(
    corrected_flow, previous_mask, current_mask, body_height, delta_time
):
    exclusion = _morphology_iterations(body_height, 0.05)
    excluded = _dilate(
        np.logical_or(previous_mask, current_mask), exclusion
    )
    usable = np.logical_not(excluded) & np.isfinite(corrected_flow).all(axis=2)
    if int(usable.sum()) < 100:
        return None
    magnitude = np.linalg.norm(corrected_flow[usable], axis=1)
    return float(np.median(magnitude) / (body_height * delta_time))


def _camera_corrected_flow(
    raw_flow,
    previous_frame,
    current_frame,
    previous_mask,
    current_mask,
    body_height,
    delta_time,
    settings,
):
    diagnostics = {
        "flow_camera_model_requested": settings.camera_correction,
        "flow_camera_model_used": "none",
        "flow_camera_ransac_success": False,
        "flow_camera_fallback_used": False,
    }
    corrected = np.asarray(raw_flow, dtype=np.float32).copy()
    if settings.camera_correction == "none":
        camera_flow = np.zeros_like(corrected)
    elif settings.camera_correction == "translation":
        translation, count = _background_translation(
            corrected, previous_mask, current_mask, body_height
        )
        camera_flow = np.broadcast_to(translation, corrected.shape)
        diagnostics.update(
            {
                "flow_camera_model_used": "translation",
                "flow_camera_background_pixel_count": int(count),
                "flow_camera_translation_x_pixels_per_frame": float(
                    translation[0]
                ),
                "flow_camera_translation_y_pixels_per_frame": float(
                    translation[1]
                ),
            }
        )
    elif settings.camera_correction in {"ransac_similarity", "ransac_affine"}:
        camera_flow, ransac = _ransac_camera_flow(
            previous_frame,
            current_frame,
            previous_mask,
            current_mask,
            body_height,
            settings,
        )
        diagnostics.update(ransac)
        if camera_flow is not None:
            diagnostics["flow_camera_model_used"] = settings.camera_correction
        else:
            translation, count = _background_translation(
                corrected, previous_mask, current_mask, body_height
            )
            camera_flow = np.broadcast_to(translation, corrected.shape)
            diagnostics.update(
                {
                    "flow_camera_model_used": (
                        "translation_fallback" if count >= 100 else "identity_fallback"
                    ),
                    "flow_camera_fallback_used": True,
                    "flow_camera_background_pixel_count": int(count),
                    "flow_camera_translation_x_pixels_per_frame": float(
                        translation[0]
                    ),
                    "flow_camera_translation_y_pixels_per_frame": float(
                        translation[1]
                    ),
                }
            )
    else:
        raise ValueError(
            "camera_correction must be one of: none, translation, "
            "ransac_similarity, ransac_affine"
        )
    corrected = corrected - camera_flow
    camera_magnitude = np.linalg.norm(camera_flow, axis=2)
    diagnostics["flow_camera_motion_body_heights_per_sec"] = float(
        np.median(camera_magnitude) / (body_height * delta_time)
    )
    diagnostics["flow_camera_residual_background_body_heights_per_sec"] = (
        _residual_background_rate(
            corrected,
            previous_mask,
            current_mask,
            body_height,
            delta_time,
        )
    )
    return corrected, diagnostics


def _forward_backward_consistency(flow, backward_flow, body_height, threshold):
    """Return pixels whose forward and backward displacements approximately cancel."""
    import cv2

    height, width = flow.shape[:2]
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    map_x = x_grid + flow[..., 0]
    map_y = y_grid + flow[..., 1]
    sampled_x = cv2.remap(
        backward_flow[..., 0], map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
    )
    sampled_y = cv2.remap(
        backward_flow[..., 1], map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
    )
    error = np.sqrt(
        (flow[..., 0] + sampled_x) ** 2 + (flow[..., 1] + sampled_y) ** 2
    )
    limit = max(0.5, float(threshold) * float(body_height))
    return np.isfinite(error) & (error <= limit), error


def _measurement_region(
    flow,
    previous_mask,
    current_mask,
    *,
    body_height,
    settings,
):
    erosion = _morphology_iterations(
        body_height, settings.erosion_body_fraction
    )
    source = _erode(previous_mask, erosion)
    if settings.region_mode == "source":
        return source
    if settings.region_mode == "intersection":
        return np.logical_and(source, current_mask)
    if settings.region_mode != "flow_consistent":
        raise ValueError(
            "region_mode must be one of: source, intersection, flow_consistent"
        )

    dilation = _morphology_iterations(
        body_height, settings.target_dilation_body_fraction
    )
    target = _dilate(current_mask, dilation)
    height, width = source.shape
    y_grid, x_grid = np.indices(source.shape)
    target_x = np.rint(x_grid + flow[..., 0]).astype(np.int64)
    target_y = np.rint(y_grid + flow[..., 1]).astype(np.int64)
    inside = np.logical_and.reduce(
        (
            target_x >= 0,
            target_x < width,
            target_y >= 0,
            target_y < height,
        )
    )
    endpoint_valid = np.zeros_like(source)
    rows, columns = np.nonzero(inside)
    endpoint_valid[rows, columns] = target[
        target_y[rows, columns], target_x[rows, columns]
    ]
    return np.logical_and(source, endpoint_valid)


def _torso_core_region(mask, valid_region, *, settings):
    """Return a central elliptical proxy for bulk torso translation."""
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return np.zeros_like(mask, dtype=bool)
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(columns.min()), int(columns.max())
    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    radius_y = max(
        1.0, float(settings.torso_radius_y_fraction) * (y1 - y0 + 1)
    )
    radius_x = max(
        1.0, float(settings.torso_radius_x_fraction) * (x1 - x0 + 1)
    )
    yy, xx = np.indices(mask.shape, dtype=np.float64)
    ellipse = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
        <= 1.0
    )
    return np.logical_and.reduce((mask.astype(bool), valid_region, ellipse))


def _vector_statistics(
    vectors, *, body_height, delta_time, active_threshold
):
    """Compute robust frame-pair statistics for total or residual flow."""
    vectors = np.asarray(vectors, dtype=np.float64)
    magnitude_pixels = np.linalg.norm(vectors, axis=1)
    magnitude = magnitude_pixels / (body_height * delta_time)
    mean_vector = np.mean(vectors, axis=0)
    mean_magnitude = float(np.mean(magnitude_pixels))
    coherence = (
        float(np.linalg.norm(mean_vector) / mean_magnitude)
        if mean_magnitude > 1e-12
        else 0.0
    )
    trimmed = np.logical_and(
        magnitude >= np.percentile(magnitude, 5),
        magnitude <= np.percentile(magnitude, 95),
    )
    return {
        "median_pixels_per_frame": float(np.percentile(magnitude_pixels, 50)),
        "median_body_heights_per_sec": float(np.percentile(magnitude, 50)),
        "p75_body_heights_per_sec": float(np.percentile(magnitude, 75)),
        "p90_body_heights_per_sec": float(np.percentile(magnitude, 90)),
        "trimmed_mean_body_heights_per_sec": float(np.mean(magnitude[trimmed])),
        "active_area_fraction": float(np.mean(magnitude >= active_threshold)),
        "directional_coherence": float(np.clip(coherence, 0.0, 1.0)),
    }


def summarize_flow_pair(
    flow,
    previous_mask,
    current_mask,
    *,
    delta_time,
    settings,
    corrected_flow=None,
    correction_diagnostics=None,
    backward_flow=None,
    body_height_override=None,
):
    """Summarize one forward flow field inside the tracked participant."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("flow must have shape HxWx2")
    if previous_mask.shape != flow.shape[:2] or current_mask.shape != flow.shape[:2]:
        raise ValueError("flow and masks must have matching spatial shapes")
    if delta_time <= 0:
        raise ValueError("delta_time must be positive")
    heights = [
        value
        for value in (_mask_height(previous_mask), _mask_height(current_mask))
        if value is not None
    ]
    if not heights:
        return {
            "flow_pair_valid": False,
            "flow_pair_failure": "empty_mask",
        }
    body_height = float(
        body_height_override
        if body_height_override is not None
        and np.isfinite(body_height_override)
        and body_height_override > 0
        else np.median(heights)
    )
    raw_flow = np.asarray(flow, dtype=np.float32)
    region = _measurement_region(
        raw_flow,
        previous_mask.astype(bool),
        current_mask.astype(bool),
        body_height=body_height,
        settings=settings,
    )
    source_iterations = _morphology_iterations(
        body_height, settings.erosion_body_fraction
    )
    source_region = _erode(previous_mask, source_iterations)
    source_pixels = int(source_region.sum())
    finite = np.isfinite(raw_flow).all(axis=2)
    region = np.logical_and(region, finite)
    forward_backward_error = None
    if settings.forward_backward_check:
        if backward_flow is None:
            raise ValueError(
                "backward_flow is required when forward_backward_check is enabled"
            )
        consistent, forward_backward_error = _forward_backward_consistency(
            raw_flow,
            np.asarray(backward_flow, dtype=np.float32),
            body_height,
            settings.forward_backward_threshold_body_fraction,
        )
        region = np.logical_and(region, consistent)
    retained_pixels = int(region.sum())
    retained_fraction = (
        float(retained_pixels / source_pixels) if source_pixels else 0.0
    )

    if corrected_flow is None:
        background = np.zeros(2, dtype=np.float32)
        background_pixels = 0
        if settings.camera_correction == "translation":
            background, background_pixels = _background_translation(
                raw_flow,
                previous_mask,
                current_mask,
                body_height,
            )
        elif settings.camera_correction != "none":
            raise ValueError(
                "RANSAC correction must be computed from the source frames"
            )
        corrected = raw_flow - background[None, None, :]
        correction_diagnostics = {
            "flow_camera_model_requested": settings.camera_correction,
            "flow_camera_model_used": settings.camera_correction,
            "flow_camera_background_pixel_count": background_pixels,
            "flow_camera_translation_x_pixels_per_frame": float(background[0]),
            "flow_camera_translation_y_pixels_per_frame": float(background[1]),
            "flow_camera_motion_body_heights_per_sec": float(
                np.linalg.norm(background) / (body_height * delta_time)
            ),
        }
    else:
        corrected = np.asarray(corrected_flow, dtype=np.float32)
        if corrected.shape != raw_flow.shape:
            raise ValueError("corrected_flow and flow must have matching shapes")
        correction_diagnostics = dict(correction_diagnostics or {})

    pair_valid = bool(
        retained_pixels >= settings.minimum_valid_pixels
        and retained_fraction >= settings.minimum_retained_pixel_fraction
    )
    base = {
        "flow_pair_valid": pair_valid,
        "flow_pair_failure": None if pair_valid else "insufficient_valid_subject_pixels",
        "body_height_pixels": body_height,
        "flow_source_pixel_count": source_pixels,
        "flow_retained_pixel_count": retained_pixels,
        "flow_retained_pixel_fraction": retained_fraction,
        "flow_forward_backward_checked": bool(settings.forward_backward_check),
    }
    base.update(correction_diagnostics)
    # Preserve legacy fields used by existing downstream tables.
    base["flow_background_pixel_count"] = int(
        base.get("flow_camera_background_pixel_count", 0) or 0
    )
    base["flow_background_translation_x_pixels_per_frame"] = float(
        base.get("flow_camera_translation_x_pixels_per_frame", 0.0) or 0.0
    )
    base["flow_background_translation_y_pixels_per_frame"] = float(
        base.get("flow_camera_translation_y_pixels_per_frame", 0.0) or 0.0
    )
    base["flow_background_translation_body_heights_per_sec"] = float(
        base.get("flow_camera_motion_body_heights_per_sec", 0.0) or 0.0
    )
    if forward_backward_error is not None and retained_pixels:
        retained_errors = forward_backward_error[region]
        base["flow_forward_backward_error_body_fraction_median"] = float(
            np.median(retained_errors) / body_height
        )
    if not pair_valid:
        return base

    vectors = corrected[region].astype(np.float64)
    total = _vector_statistics(
        vectors,
        body_height=body_height,
        delta_time=delta_time,
        active_threshold=settings.active_threshold_body_heights_per_sec,
    )
    base.update(
        {
            "flow_pixel_median_pixels_per_frame": total["median_pixels_per_frame"],
            "flow_pixel_median_body_heights_per_sec": total["median_body_heights_per_sec"],
            "flow_pixel_p75_body_heights_per_sec": total["p75_body_heights_per_sec"],
            "flow_pixel_p90_body_heights_per_sec": total["p90_body_heights_per_sec"],
            "flow_pixel_trimmed_mean_body_heights_per_sec": total["trimmed_mean_body_heights_per_sec"],
            "flow_active_area_fraction": total["active_area_fraction"],
            "flow_directional_coherence": total["directional_coherence"],
        }
    )

    torso_region = _torso_core_region(previous_mask, region, settings=settings)
    torso_pixels = int(torso_region.sum())
    torso_valid = bool(
        settings.torso_correction
        and torso_pixels >= int(settings.torso_minimum_pixels)
    )
    base["flow_torso_correction_requested"] = bool(settings.torso_correction)
    base["flow_torso_core_pixel_count"] = torso_pixels
    base["flow_torso_correction_valid"] = torso_valid
    if torso_valid:
        torso_vector = np.median(corrected[torso_region], axis=0).astype(
            np.float64
        )
        residual = _vector_statistics(
            vectors - torso_vector[None, :],
            body_height=body_height,
            delta_time=delta_time,
            active_threshold=settings.active_threshold_body_heights_per_sec,
        )
        base.update(
            {
                "flow_torso_translation_x_pixels_per_frame": float(torso_vector[0]),
                "flow_torso_translation_y_pixels_per_frame": float(torso_vector[1]),
                "flow_torso_translation_body_heights_per_sec": float(
                    np.linalg.norm(torso_vector) / (body_height * delta_time)
                ),
                "flow_pixel_median_torso_corrected_body_heights_per_sec": residual["median_body_heights_per_sec"],
                "flow_pixel_p75_torso_corrected_body_heights_per_sec": residual["p75_body_heights_per_sec"],
                "flow_pixel_p90_torso_corrected_body_heights_per_sec": residual["p90_body_heights_per_sec"],
                "flow_pixel_trimmed_mean_torso_corrected_body_heights_per_sec": residual["trimmed_mean_body_heights_per_sec"],
                "flow_torso_corrected_active_area_fraction": residual["active_area_fraction"],
                "flow_torso_corrected_directional_coherence": residual["directional_coherence"],
            }
        )
    return base


def _bout_measurements(rows, settings):
    core_rows = [row for row in rows if row.get("is_core_pair")]
    valid_rows = [row for row in core_rows if row.get("flow_pair_valid")]
    torso_valid_rows = [
        row for row in valid_rows if row.get("flow_torso_correction_valid")
    ]
    n_core = len(core_rows)
    n_valid = len(valid_rows)
    valid_fraction = float(n_valid / n_core) if n_core else 0.0

    def values(name):
        return [
            row[name]
            for row in valid_rows
            if row.get(name) is not None and np.isfinite(row[name])
        ]

    result = {
        "optical_flow_backend": settings.backend,
        "optical_flow_camera_correction": settings.camera_correction,
        "optical_flow_region_mode": settings.region_mode,
        "optical_flow_n_core_pairs": n_core,
        "optical_flow_n_valid_pairs": n_valid,
        "optical_flow_valid_pair_fraction": valid_fraction,
        "optical_flow_measurement_reliable": bool(
            n_valid >= 2
            and valid_fraction >= settings.minimum_valid_pair_fraction
        ),
        "optical_flow_torso_valid_pair_fraction": (
            float(len(torso_valid_rows) / n_valid) if n_valid else 0.0
        ),
        "optical_flow_torso_measurement_reliable": bool(
            len(torso_valid_rows) >= 2
            and valid_fraction >= settings.minimum_valid_pair_fraction
            and len(torso_valid_rows) / max(1, n_valid)
            >= settings.minimum_valid_pair_fraction
        ),
        "optical_flow_ransac_success_fraction": (
            float(
                np.mean(
                    [
                        bool(row.get("flow_camera_ransac_success"))
                        for row in valid_rows
                    ]
                )
            )
            if valid_rows
            and settings.camera_correction
            in {"ransac_similarity", "ransac_affine"}
            else None
        ),
        "optical_flow_camera_fallback_fraction": (
            float(
                np.mean(
                    [
                        bool(row.get("flow_camera_fallback_used"))
                        for row in valid_rows
                    ]
                )
            )
            if valid_rows
            and settings.camera_correction
            in {"ransac_similarity", "ransac_affine"}
            else None
        ),
    }
    source_names = (
        "flow_pixel_median_body_heights_per_sec",
        "flow_pixel_p75_body_heights_per_sec",
        "flow_pixel_p90_body_heights_per_sec",
        "flow_pixel_trimmed_mean_body_heights_per_sec",
        "flow_active_area_fraction",
        "flow_directional_coherence",
        "flow_retained_pixel_fraction",
        "flow_background_translation_body_heights_per_sec",
        "flow_camera_residual_background_body_heights_per_sec",
        "flow_forward_backward_error_body_fraction_median",
        "flow_torso_core_pixel_count",
        "flow_torso_translation_body_heights_per_sec",
        "flow_pixel_median_torso_corrected_body_heights_per_sec",
        "flow_pixel_p75_torso_corrected_body_heights_per_sec",
        "flow_pixel_p90_torso_corrected_body_heights_per_sec",
        "flow_pixel_trimmed_mean_torso_corrected_body_heights_per_sec",
        "flow_torso_corrected_active_area_fraction",
        "flow_torso_corrected_directional_coherence",
    )
    for name in source_names:
        result["optical_%s_median" % name] = _safe_summary(
            values(name), "median"
        )
        result["optical_%s_temporal_p90" % name] = _safe_summary(
            values(name), "p90"
        )
    return result


def analyze_optical_flow_arrays(
    frames,
    masks,
    timestamps,
    *,
    settings=None,
    estimator=None,
    valid_mask=None,
):
    """Run the production optical-flow estimator on in-memory arrays.

    This is the shared API used by controlled Skeleton Lab validation. Real
    bouts use :func:`analyze_optical_flow_episode`, which delegates to the same
    flow, camera-correction, mask-consistency, and torso-correction primitives.
    """
    settings = settings or OpticalFlowSettings()
    frames = np.asarray(frames)
    masks = np.asarray(masks, dtype=bool)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (time, height, width, 3)")
    if masks.ndim != 3 or masks.shape != frames.shape[:3]:
        raise ValueError("masks must match the frame time and spatial dimensions")
    if len(timestamps) != len(frames) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must match frames and be strictly increasing")
    if valid_mask is None:
        valid_mask = masks.reshape(len(masks), -1).any(axis=1)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(valid_mask) != len(frames):
        raise ValueError("valid_mask must contain one value per frame")
    if estimator is None:
        estimator = make_flow_estimator(settings)

    smoothed_heights = _smooth_mask_heights(
        masks, valid_mask, settings.body_height_smoothing_window
    )
    rows = []
    for index in range(len(frames) - 1):
        delta_time = float(timestamps[index + 1] - timestamps[index])
        row = {
            "frame_index": index,
            "next_frame_index": index + 1,
            "delta_time_sec": delta_time,
            "is_core_pair": True,
            "flow_pair_valid": False,
        }
        if not valid_mask[index] or not valid_mask[index + 1]:
            row["flow_pair_failure"] = "missing_or_unobservable_mask"
            rows.append(row)
            continue
        flow = estimator.estimate(frames[index], frames[index + 1])
        backward = (
            estimator.estimate(frames[index + 1], frames[index])
            if settings.forward_backward_check
            else None
        )
        body_height = float(np.median(smoothed_heights[index : index + 2]))
        corrected, diagnostics = _camera_corrected_flow(
            flow,
            frames[index],
            frames[index + 1],
            masks[index],
            masks[index + 1],
            body_height,
            delta_time,
            settings,
        )
        row.update(
            summarize_flow_pair(
                flow,
                masks[index],
                masks[index + 1],
                delta_time=delta_time,
                settings=settings,
                corrected_flow=corrected,
                correction_diagnostics=diagnostics,
                backward_flow=backward,
                body_height_override=body_height,
            )
        )
        rows.append(row)
    return {"measurements": _bout_measurements(rows, settings), "frame_pairs": rows}


def _read_video(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video: %s" % path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("Video contained no decodable frames: %s" % path)
    return frames


def load_saved_masks(path):
    """Load original-resolution SAM masks and timing arrays from one run."""
    with np.load(path) as saved:
        if "source_masks_packed" not in saved.files:
            raise ValueError(
                "The saved run has no original-resolution source masks. "
                "Rerun the SAM+DINO stage with the current pipeline."
            )
        width = int(np.asarray(saved["source_mask_width"]).item())
        masks = np.unpackbits(
            saved["source_masks_packed"], axis=-1
        )[..., :width].astype(bool)
        timestamps = saved["timestamps_sec"].astype(np.float64)
        core_mask = saved["core_mask"].astype(bool)
        valid = (
            saved["mask_valid"].astype(bool)
            if "mask_valid" in saved.files
            else masks.reshape(len(masks), -1).any(axis=1)
        )
    return masks, timestamps, core_mask, valid


def resolve_source_video(run_dir, episodes_root = None):
    summary = _read_json(run_dir / "subject_dino_summary.json")
    candidates = []
    if summary.get("source_cached_video"):
        candidates.append(Path(summary["source_cached_video"]).expanduser())
    if summary.get("source_episode_dir"):
        candidates.extend(
            sorted(
                (Path(summary["source_episode_dir"]).expanduser() / "assets").glob(
                    "*_raw.mp4"
                )
            )
        )
    if episodes_root is not None:
        candidates.extend(
            sorted((episodes_root / run_dir.name / "assets").glob("*_raw.mp4"))
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve the cached *_raw.mp4 for %s" % run_dir.name
    )


def _write_rows(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _flow_color_image(frame, flow, region, value_limit):
    import cv2

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame)
    hsv[..., 0] = np.mod(angle * 90.0 / np.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        255.0 * magnitude / max(value_limit, 1e-6), 0, 255
    ).astype(np.uint8)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    output = frame.copy()
    output[region] = cv2.addWeighted(
        frame, 0.25, color, 0.75, 0
    )[region]
    return output


def analyze_optical_flow_episode(
    run_dir,
    *,
    settings,
    estimator = None,
    episodes_root = None,
    create_figure = True,
    create_video = False,
    force = False,
):
    """Measure visual motion using an existing completed SAM2 run."""
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "optical_flow_summary.json"
    requested_settings = asdict(settings)
    existing = _read_json(summary_path)
    if (
        summary_path.is_file()
        and not force
        and existing.get("schema_version") == "prism_masked_optical_flow.v1"
        and existing.get("settings") == requested_settings
    ):
        return existing

    source_video = resolve_source_video(run_dir, episodes_root=episodes_root)
    frames = _read_video(source_video)
    masks, timestamps, core_mask, mask_valid = load_saved_masks(
        run_dir / "sam2_masks_fixed_crop.npz"
    )
    lengths = {
        len(frames),
        len(masks),
        len(timestamps),
        len(core_mask),
        len(mask_valid),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Video, masks, timestamps, core mask, and validity mask have "
            "different frame counts: %s" % sorted(lengths)
        )
    if tuple(frames[0].shape[:2]) != tuple(masks.shape[1:]):
        raise ValueError(
            "Video frame shape %s does not match source-mask shape %s"
            % (frames[0].shape[:2], masks.shape[1:])
        )
    if estimator is None:
        estimator = make_flow_estimator(settings)
    smoothed_heights = _smooth_mask_heights(
        masks,
        mask_valid,
        settings.body_height_smoothing_window,
    )

    video_writer = None
    rows = []
    for index in range(len(frames) - 1):
        delta_time = float(timestamps[index + 1] - timestamps[index])
        is_core_pair = bool(core_mask[index] and core_mask[index + 1])
        row = {
            "frame_index": index,
            "next_frame_index": index + 1,
            "timestamp_sec": float(timestamps[index]),
            "next_timestamp_sec": float(timestamps[index + 1]),
            "delta_time_sec": delta_time,
            "is_core_pair": is_core_pair,
            "previous_mask_valid": bool(mask_valid[index]),
            "current_mask_valid": bool(mask_valid[index + 1]),
            "flow_pair_valid": False,
            "flow_pair_failure": None,
        }
        if not is_core_pair:
            row["flow_pair_failure"] = "outside_annotated_core"
            rows.append(row)
            continue
        if not mask_valid[index] or not mask_valid[index + 1]:
            row["flow_pair_failure"] = "missing_or_unobservable_mask"
            rows.append(row)
            continue
        if delta_time <= 0:
            row["flow_pair_failure"] = "nonpositive_time_increment"
            rows.append(row)
            continue

        flow = estimator.estimate(frames[index], frames[index + 1])
        backward_flow = None
        if settings.forward_backward_check:
            backward_flow = estimator.estimate(
                frames[index + 1], frames[index]
            )
        body_height = float(
            np.median(smoothed_heights[index : index + 2])
        )
        corrected_flow, correction_diagnostics = _camera_corrected_flow(
            flow,
            frames[index],
            frames[index + 1],
            masks[index],
            masks[index + 1],
            body_height,
            delta_time,
            settings,
        )
        pair = summarize_flow_pair(
            flow,
            masks[index],
            masks[index + 1],
            delta_time=delta_time,
            settings=settings,
            corrected_flow=corrected_flow,
            correction_diagnostics=correction_diagnostics,
            backward_flow=backward_flow,
            body_height_override=body_height,
        )
        row.update(pair)
        rows.append(row)

        if create_video and pair.get("flow_pair_valid"):
            import cv2

            height = float(pair["body_height_pixels"])
            region = _measurement_region(
                flow,
                masks[index],
                masks[index + 1],
                body_height=height,
                settings=settings,
            )
            if settings.forward_backward_check and backward_flow is not None:
                consistent, _ = _forward_backward_consistency(
                    flow,
                    backward_flow,
                    height,
                    settings.forward_backward_threshold_body_fraction,
                )
                region = np.logical_and(region, consistent)
            values = np.linalg.norm(corrected_flow[region], axis=1)
            limit = float(np.percentile(values, 95)) if len(values) else 1.0
            rendered = _flow_color_image(
                frames[index], corrected_flow, region, value_limit=limit
            )
            cv2.putText(
                rendered,
                "p75 %.3f body heights/s"
                % pair["flow_pixel_p75_body_heights_per_sec"],
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(run_dir / "optical_flow_qc.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    1.0 / max(delta_time, 1e-6),
                    (rendered.shape[1], rendered.shape[0]),
                )
                if not video_writer.isOpened():
                    raise RuntimeError("Could not create optical_flow_qc.mp4")
            video_writer.write(rendered)
    if video_writer is not None:
        video_writer.release()

    measurements = _bout_measurements(rows, settings)
    source_summary = _read_json(run_dir / "subject_dino_summary.json")
    summary = {
        "schema_version": "prism_masked_optical_flow.v1",
        "episode_id": run_dir.name,
        "episode": source_summary.get("episode") or {},
        "fps": source_summary.get("fps"),
        "source_video": str(source_video),
        "source_masks": str(run_dir / "sam2_masks_fixed_crop.npz"),
        "settings": requested_settings,
        "measurements": measurements,
    }
    _write_rows(run_dir / "optical_flow_frames.csv", rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    if create_figure:
        from .plotting.optical_flow import plot_flow_trace

        plot_flow_trace(run_dir / "optical_flow_trace.png", rows)
    return summary
