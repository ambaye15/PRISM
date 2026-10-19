import numpy as np

from .masks import mask_geometry, require_nonempty_masks, validate_sequence


def interpolation_operators(timestamps, start_times, d, tau_sec):

    n_frames = len(timestamps)
    rows = np.arange(len(start_times))
    operators = []
    for delay in range(d + 1):
        query = np.clip(start_times + delay * tau_sec, timestamps[0], timestamps[-1])
        right = np.searchsorted(timestamps, query, side="right")
        right = np.clip(right, 1, n_frames - 1).astype(int)
        left = right - 1
        alpha = np.clip(
            (query - timestamps[left]) / (timestamps[right] - timestamps[left]), 0.0, 1.0
        )
        operator = np.zeros((len(start_times), n_frames), dtype=np.float64)
        operator[rows, left] += 1.0 - alpha
        operator[rows, right] += alpha
        operators.append(operator)
    return operators

def sliding_window_distances(masks, timestamps, *, period_sec, d = 23, n_points = 300):
    masks, timestamps = validate_sequence(masks, timestamps)
    require_nonempty_masks(masks)
    if d < 1 or n_points < 8 or not np.isfinite(period_sec) or period_sec <= 0:
        raise ValueError("require d>=1, n_points>=8, and a positive period")
    tau = float(period_sec) / (d + 1)
    span = d * tau
    latest = float(timestamps[-1] - span)
    if latest <= timestamps[0]:
        raise ValueError("the mask sequence is too short for this period and d")
    starts = np.linspace(float(timestamps[0]), latest, n_points)
    geometry = mask_geometry(masks)
    gram = geometry["gram"]
    operators = interpolation_operators(timestamps, starts, d, tau)
    raw_dot = np.zeros((n_points, n_points), dtype=np.float64)
    summed = np.zeros_like(operators[0])
    for operator in operators:
        raw_dot += operator @ gram @ operator.T
        summed += operator
    raw_dot = 0.5 * (raw_dot + raw_dot.T)
    raw_norms = np.diag(raw_dot)
    raw_squared = np.maximum(raw_norms[:, None] + raw_norms[None, :] - 2.0 * raw_dot, 0.0)
    scale = np.sqrt(raw_squared / max(geometry["median_area"] * (d + 1), 1.0))

    temporal_mean_dot = (summed @ gram @ summed.T) / (d + 1)
    centered_dot = raw_dot - temporal_mean_dot
    centered_dot = 0.5 * (centered_dot + centered_dot.T)
    centered_norms = np.maximum(np.diag(centered_dot), 0.0)

    sphere_valid = np.isfinite(centered_norms) & (centered_norms > 1e-12)

    if sphere_valid.sum() < 8:
        raise ValueError(
            "fewer than 8 nonconstant sliding-window points remain "
            "for sphere normalization"
        )

    normalized_dot = centered_dot[np.ix_(sphere_valid, sphere_valid)]
    normalized_norms = centered_norms[sphere_valid]

    denominator = np.sqrt(
        np.outer(normalized_norms, normalized_norms)
    )
    cosine = np.clip(normalized_dot / denominator, -1.0, 1.0)
    normalized = np.sqrt(
        np.maximum(2.0 - 2.0 * cosine, 0.0)
    )

    scale[scale < 1e-7] = 0.0
    normalized[normalized < 1e-7] = 0.0
    np.fill_diagonal(scale, 0.0)
    np.fill_diagonal(normalized, 0.0)

    return {
        "start_times_sec": starts,
        "normalized_start_times_sec": starts[sphere_valid],
        "tau_sec": tau,
        "window_span_sec": span,
        "scale_distance": scale,
        "normalized_distance": normalized,
        "pre_sphere_norm": np.sqrt(normalized_norms),
        "sphere_valid_mask": sphere_valid,
        "n_sphere_points": int(sphere_valid.sum()),
        "sphere_valid_fraction": float(sphere_valid.mean()),
    }
